use burn::{
    nn::{
        Linear, LinearConfig,
        loss::{MseLoss, Reduction},
    },
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    prelude::*,
    tensor::{BasicOps, TensorKind, activation::relu, backend::AutodiffBackend},
    train::TrainOutput,
};
use strum::IntoEnumIterator;

use crate::{
    game::{
        card::Card,
        game::{Game, Score},
    },
    training::{
        data::{ReplayBuffer, ReplayItem},
        simulation::Action,
    },
};

use super::{Model, Train, Valid};

#[derive(Config, Copy)]
pub struct Weights {
    #[config(default = 1e-4)]
    pub lr: f32,

    #[config(default = 0.99)]
    pub gamma: f32,

    #[config(default = 0.0)]
    pub eps: f32,
}

#[derive(Clone, Debug)]
pub struct State {
    pub game: Game,
    pub player: usize,
}

impl State {
    const SIZE: usize = 3;

    pub fn new(game: Game) -> Self {
        Self { game, player: 0 }
    }

    pub fn normalize(&self) -> [f32; Self::SIZE] {
        let player_score = self.game.player_score(self.player);
        let dealer_upcard = self.game.dealer_upcard();

        Self::raw_normalize(&player_score, &dealer_upcard)
    }

    pub fn raw_normalize(player_score: &Score, dealer_upcard: &Card) -> [f32; Self::SIZE] {
        let player_soft = match player_score {
            Score::Hard(_) => false as u8 as f32,
            Score::Soft(_) => true as u8 as f32,
        };

        [
            player_score.value() as f32 / 21.0,
            player_soft as u8 as f32,
            dealer_upcard.rank.value() as f32 / 11.0,
        ]
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = 128)]
    hidden_size: usize,
}

#[derive(Module, Debug)]
pub struct Dqn<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    value: Linear<B>,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Dqn<B> {
        Dqn::new(self, device)
    }
}

impl<B> Dqn<B>
where
    B: Backend,
{
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        let fc1 = LinearConfig::new(Self::input_size(), config.hidden_size).init(device);
        let fc2 = LinearConfig::new(config.hidden_size, config.hidden_size).init(device);
        let value = LinearConfig::new(config.hidden_size, Self::output_size()).init(device);

        Self { fc1, fc2, value }
    }
}

impl<B> Model<B> for Dqn<B>
where
    B: Backend,
{
    type State = State;

    fn input_size() -> usize {
        State::SIZE
    }

    fn output_size() -> usize {
        Action::iter().len() - 1
    }

    fn normalize(state: &[State], device: &B::Device) -> Tensor<B, 2> {
        let tensors = state
            .iter()
            .map(|state| Tensor::<B, 1>::from_floats(state.normalize(), device).reshape([1, -1]))
            .collect();

        Tensor::<B, 2>::cat(tensors, 0)
    }

    fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(state);
        let x = relu(x);
        let x = self.fc2.forward(x);
        let x = relu(x);
        let x = self.value.forward(x);

        x
    }
}

#[derive(Config)]
pub struct LearnerConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
}

impl LearnerConfig {
    pub fn init<B>(self, device: B::Device) -> Learner<B, OptimizerAdaptor<Adam, Dqn<B>, B>>
    where
        B: AutodiffBackend,
    {
        Learner::new(&self, device)
    }
}

pub struct Learner<B, O>
where
    B: AutodiffBackend,
    O: Optimizer<Dqn<B>, B>,
{
    model: Dqn<B>,
    optimizer: O,
    device: B::Device,
}

impl<B> Learner<B, OptimizerAdaptor<Adam, Dqn<B>, B>>
where
    B: AutodiffBackend,
{
    pub fn new(config: &LearnerConfig, device: B::Device) -> Self {
        Self {
            model: config.model.init(&device),
            optimizer: config.optimizer.init().to_owned(),
            device: device,
        }
    }

    pub fn model(&self) -> &Dqn<B> {
        &self.model
    }
}

fn collate<B: Backend, const D: usize, T: TensorKind<B>>(
    sample: &[&ReplayItem<B>],
    filter: fn(&ReplayItem<B>) -> Tensor<B, D, T>,
) -> Tensor<B, D, T>
where
    T: BasicOps<B>,
{
    let data = sample.iter().map(|item| filter(item)).collect();
    Tensor::<B, D, T>::cat(data, 0)
}

impl<B, O> Train<Tensor<B, 1>> for Learner<B, O>
where
    B: AutodiffBackend,
    O: Optimizer<Dqn<B>, B>,
{
    type Batch = ReplayBuffer<B>;
    type Weights = Weights;

    fn train_step(
        &self,
        batch: &ReplayBuffer<B>,
        weights: &Weights,
        iteration: usize,
    ) -> TrainOutput<Tensor<B, 1>> {
        let sample = batch.sample(iteration);

        let state = collate(&sample, |item| item.state.clone());
        let action = collate(&sample, |item| item.action.clone());
        let reward = collate(&sample, |item| item.reward.clone());
        let terminal = collate(&sample, |item| item.terminal.clone());
        let next_state = collate(&sample, |item| item.next_state.clone());

        let q = self.model.forward(state);
        let chosen_q = q.clone().gather(1, action.clone());

        let next_filter = Tensor::<B, 1, Int>::from_data(
            [usize::from(Action::Hit), usize::from(Action::Stand)],
            &self.device,
        );

        let next_q = self.model.forward(next_state).detach();
        let max_next_q = next_q.select(1, next_filter).max_dim(1);

        let pending = -(terminal - 1.0);
        let target = reward + pending * weights.gamma * max_next_q;

        let loss = MseLoss::new().forward(chosen_q, target, Reduction::Mean);

        TrainOutput::new(&self.model, loss.backward(), loss)
    }

    fn fit(mut self, grads: GradientsParams, weights: &Self::Weights) -> Self {
        self.model = self.optimizer.step(weights.lr as f64, self.model, grads);
        self
    }
}

impl<B, O> Valid<Tensor<B, 1>> for Learner<B, O>
where
    B: AutodiffBackend,
    O: Optimizer<Dqn<B>, B>,
{
    type Batch = ReplayBuffer<B>;
    type Weights = Weights;

    fn valid_step(
        &self,
        batch: &Self::Batch,
        weights: &Self::Weights,
        iteration: usize,
    ) -> TrainOutput<Tensor<B, 1>> {
        self.train_step(batch, weights, iteration)
    }
}
