use core::f32;
use std::{cell::RefCell, ops::Deref, sync::Mutex};

use burn::{
    nn::{Linear, LinearConfig},
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    prelude::*,
    record::Record,
    tensor::{
        activation::{relu, softmax},
        backend::AutodiffBackend,
        cast::ToElement,
    },
    train::TrainOutput,
};
use num_enum::IntoPrimitive;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, ParallelBridge, ParallelIterator,
};
use strum::{EnumIter, IntoEnumIterator};

use crate::game::{
    card::Card,
    game::{Game, Outcome, Score},
};

use super::simulation::{Action, Simulation};

#[derive(IntoPrimitive, EnumIter, Clone, Copy, Debug)]
#[repr(usize)]
enum Input {
    PlayerScore,
    IsSoft,
    DealerUpCard,
}

#[derive(Clone, Copy, Debug)]
pub struct State {
    pub player_score: Score,
    pub dealer_upcard: Card,
}

#[derive(Config)]
pub struct ModelConfig {
    #[config(default = 128)]
    hidden_size: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> Model<B> {
        Model::new(self, device)
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    policy: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn new(config: ModelConfig, device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(Self::input_size(), config.hidden_size).init(device),
            fc2: LinearConfig::new(config.hidden_size, config.hidden_size).init(device),
            policy: LinearConfig::new(config.hidden_size, Self::output_size()).init(device),
        }
    }

    pub fn normalize(state: &[State], device: &B::Device) -> Tensor<B, 2> {
        let data: Vec<f32> = state
            .par_iter()
            .map(|state| {
                [
                    state.player_score.value() as f32 / 21.0,
                    match state.player_score {
                        Score::Hard(_) => false as u8 as f32,
                        Score::Soft(_) => true as u8 as f32,
                    },
                    state.dealer_upcard.rank.value() as f32 / 21.0,
                ]
            })
            .flatten()
            .collect();

        Tensor::<B, 1>::from_data(data.as_slice(), device)
            .reshape([state.len(), Self::input_size()])
    }

    pub fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(state.clone());
        let x = relu(x.clone());
        let x = self.fc2.forward(x.clone());
        let x = self.policy.forward(x.clone());
        let x = x.clamp(-10, 10);

        softmax(x, 1)
    }

    pub fn input_size() -> usize {
        Input::iter().len()
    }

    pub fn output_size() -> usize {
        Action::iter().len() - 1 // don't evaluate split... yet
    }
}

pub struct LearnerConfig {
    model: ModelConfig,
    optimizer: AdamConfig,
}

impl LearnerConfig {
    pub fn init<B>(self, device: B::Device) -> Learner<B, OptimizerAdaptor<Adam, Model<B>, B>>
    where
        B: AutodiffBackend,
    {
        Learner::new(self.model.init(&device), self.optimizer.init(), device)
    }
}

pub struct Learner<B, O>
where
    B: AutodiffBackend,
    O: Optimizer<Model<B>, B>,
{
    model: Model<B>,
    optimizer: O,
    device: B::Device,
}

impl<B, O> Learner<B, O>
where
    B: AutodiffBackend,
    O: Optimizer<Model<B>, B>,
{
    pub fn new(model: Model<B>, optimizer: O, device: B::Device) -> Self {
        Self {
            model: model.clone(),
            optimizer,
            device,
        }
    }

    pub fn model(&self) -> Model<B> {
        self.model.clone()
    }

    pub fn fit(mut self, item: TrainOutput<Tensor<B, 1>>, lr: f32) -> Self {
        self.model = self.optimizer.step(lr as f64, self.model, item.grads);
        self
    }

    pub fn train_step(&self, env: &[Game], state: Tensor<B, 2>) -> TrainOutput<Tensor<B, 1>> {
        self.step(env, state, true)
    }

    pub fn valid_step(&self, env: &[Game], state: Tensor<B, 2>) -> TrainOutput<Tensor<B, 1>> {
        self.step(env, state, false)
    }

    fn step(
        &self,
        env: &[Game],
        state: Tensor<B, 2>,
        train_step: bool,
    ) -> TrainOutput<Tensor<B, 1>> {
        let batch_probs = self.model.forward(state);

        let par_batch: Vec<(Game, Tensor<B, 1>, &B::Device, Mutex<RefCell<Model<B>>>)> = env
            .iter()
            .enumerate()
            .map(|(ndx, game)| {
                let select = Tensor::<B, 1, Int>::from_data([ndx], &self.device);
                (
                    game.clone(),
                    batch_probs.clone().select(0, select).squeeze(0),
                    &self.device,
                    Mutex::new(RefCell::new(self.model.clone())),
                )
            })
            .collect();

        let loss: Vec<Tensor<B, 1>> = par_batch
            .par_iter()
            .map(|(game, probs, device, model)| {
                let mut game = game.clone();
                let mut probs = probs.clone();
                let mut history: Vec<Tensor<B, 1>> = Vec::new();
                let mut rewards: Vec<f32> = Vec::new();

                let mut first_hand = true;
                loop {
                    let filter: Vec<usize> = if first_hand {
                        vec![
                            Action::Hit,
                            Action::Stand,
                            Action::Double,
                            Action::Surrender,
                        ]
                    } else {
                        vec![Action::Hit, Action::Stand]
                    }
                    .iter()
                    .map(|&action| usize::from(action))
                    .collect();

                    let action = if train_step {
                        Self::sample_categorical(probs.clone(), &filter, device)
                    } else {
                        Self::sample_max(probs.clone(), &filter, device)
                    };

                    game = Simulation::new(game).forward(0, action);

                    let select = Tensor::<B, 1, Int>::from_data([usize::from(action)], device);
                    history.push(probs.select(0, select));

                    if !game.player_active(0) {
                        game.end();
                        break;
                    }

                    // give a small incentive to hit, helps bias against "stand-only" policies
                    rewards.push(0.1);

                    let state = Model::<B>::normalize(
                        &[State {
                            player_score: game.player_score(0),
                            dealer_upcard: game.dealer_upcard(),
                        }],
                        &device,
                    );
                    probs = model.lock().unwrap().borrow().forward(state).squeeze(0);

                    first_hand = false;
                }

                let terminal_reward = match game.player_outcome(0).unwrap() {
                    Outcome::PlayerWin(amount) => amount,
                    Outcome::DealerWin(amount) => -amount,
                    Outcome::Push => 0.0,
                };
                rewards.push(terminal_reward);

                let gamma: f32 = 0.99;
                let discounted_rewards: Vec<f32> = rewards
                    .iter()
                    .rev()
                    .enumerate()
                    .map(|(step, reward)| gamma.powf(step as f32) * terminal_reward)
                    .rev()
                    .collect();

                assert_eq!(discounted_rewards.len(), history.len());

                let normalized_rewards = if discounted_rewards.len() > 1 {
                    let len = discounted_rewards.len() as f32;
                    let mean: f32 = discounted_rewards.iter().sum::<f32>() / len;
                    let variance: f32 = discounted_rewards
                        .iter()
                        .map(|r| (r - mean).powf(2.0))
                        .sum::<f32>()
                        / len;
                    let std = variance.sqrt().max(1e-6);

                    discounted_rewards
                        .iter()
                        .map(|r| (r - mean) / std)
                        .collect()
                    // discounted_rewards
                } else {
                    discounted_rewards
                };

                let rewards = Tensor::<B, 1>::from_data(normalized_rewards.as_slice(), &device);

                let probs = Tensor::cat(history, 0);
                let log_probs = probs.clone().log();

                let entropy = if train_step {
                    (probs.clone() * log_probs.clone())
                        .sum()
                        .mean()
                        .into_scalar()
                        .to_f32()
                } else {
                    0.0
                };

                -(log_probs * rewards).mean()
                // -(log_probs * rewards).mean()
                // -(log_probs * terminal_reward).mean() + entropy * 0.01
            })
            .collect();

        let loss = Tensor::cat(loss, 0);

        TrainOutput {
            grads: GradientsParams::from_grads(loss.backward(), &self.model),
            item: loss,
        }
    }

    fn sample_categorical(probs: Tensor<B, 1>, filter: &[usize], device: &B::Device) -> Action {
        let filter = Tensor::<B, 1, Int>::from_data(filter, device);
        let filtered_probs = probs.select(0, filter);

        let data: Vec<f32> = filtered_probs.to_data().to_vec().unwrap();
        let action = WeightedIndex::new(data).unwrap().sample(&mut rand::rng());

        Action::try_from(action).unwrap()
    }

    fn sample_max(probs: Tensor<B, 1>, filter: &[usize], device: &B::Device) -> Action {
        let filter = Tensor::<B, 1, Int>::from_data(filter, device);
        let filtered_probs = probs.select(0, filter);

        let action = filtered_probs.argmax(0).into_scalar().to_usize();

        Action::try_from(action).unwrap()
    }
}
