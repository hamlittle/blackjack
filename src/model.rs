use burn::{
    nn::{
        Linear, LinearConfig, Relu,
        loss::{MseLoss, Reduction},
    },
    prelude::*,
    tensor::{backend::AutodiffBackend, cast::ToElement},
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep},
};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use strum::{EnumIter, IntoEnumIterator};

use crate::{
    data::GameBatch,
    game::{Game, Outcome, Score},
};

#[derive(TryFromPrimitive, EnumIter)]
#[repr(usize)]
pub enum Action {
    Hit = 0,
    Stand = 1,
    Double = 2,
    Split = 3,
    Surrender = 4,
}

#[derive(IntoPrimitive, EnumIter)]
#[repr(usize)]
enum Input {
    PlayerScore = 0,
    IsSoft = 1,
    DealerUpCard = 2,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = 64)]
    hidden_size: usize,
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    activation: Relu,
    advantage: Linear<B>,
    value: Linear<B>,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model::new(self, device)
    }
}

impl<B: Backend> Model<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        let num_inputs = Input::iter().len();
        let num_outputs = Action::iter().len();

        let fc1 = LinearConfig::new(num_inputs, config.hidden_size).init(device);
        let fc2 = LinearConfig::new(config.hidden_size, config.hidden_size).init(device);
        let activation = Relu::new();
        let advantage = LinearConfig::new(config.hidden_size, num_outputs).init(device);
        let value = LinearConfig::new(config.hidden_size, 1).init(device);

        Self {
            fc1,
            fc2,
            activation,
            advantage,
            value,
        }
    }

    pub fn normalize(game: &Game, device: &B::Device) -> Tensor<B, 2> {
        let len = Input::iter().len();
        let data: Vec<B::FloatElem> = Input::iter()
            .map(|input| match input {
                Input::PlayerScore => game.player_score(0).value() as f32 / 21.0,
                Input::IsSoft => match game.player_score(0) {
                    Score::Soft(_) => true as u8 as f32,
                    Score::Hard(_) => false as u8 as f32,
                },
                Input::DealerUpCard => game.dealer_upcard().rank.value() as f32,
            })
            .map(|elem| B::FloatElem::from_elem(elem))
            .collect();

        let data = TensorData::new(data, [1, len]);

        Tensor::from_data(data, device)
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(input);
        let x = self.fc2.forward(x);
        let x = self.activation.forward(x);
        let advantage = self.advantage.forward(x.clone());
        let value = self.value.forward(x);

        value + (advantage.clone() - advantage.mean_dim(1))
    }

    pub fn forward_step(&self, batch: GameBatch<B>) -> RegressionOutput<B> {
        let output = self.forward(batch.input.clone());

        let action = output
            .clone()
            .argmax(1)
            .to_data()
            .to_vec::<B::IntElem>()
            .unwrap()[0]
            .to_usize();

        let mut game = batch.game.clone();
        let players = play_game(&mut game, action.try_into().unwrap());

        let reward = compute_reward(self, &output.device(), &game, &players);
        let targets = update_targets(output.clone(), action, reward);
        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Reduction::Mean);

        RegressionOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<GameBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, item: GameBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_step(item);
        let gradients = item.loss.backward();

        TrainOutput::new(self, gradients, item)
    }
}

impl<B: Backend> ValidStep<GameBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, item: GameBatch<B>) -> RegressionOutput<B> {
        self.forward_step(item)
    }
}

fn play_game(game: &mut Game, action: Action) -> Vec<usize> {
    match Action::try_from(action).unwrap() {
        Action::Hit => {
            game.player_hit(0);
            vec![0]
        }
        Action::Stand => {
            game.player_stand(0);
            vec![0]
        }
        Action::Double => {
            game.player_double(0);
            vec![0]
        }
        Action::Split => {
            let split_play = game.player_split(0);
            vec![0, split_play]
        }
        Action::Surrender => {
            game.player_surrender(0);
            vec![0]
        }
    }
}

fn compute_reward<B: Backend>(
    model: &Model<B>,
    device: &B::Device,
    game: &Game,
    players: &[usize],
) -> f32 {
    const GAMMA: f32 = 0.99;

    let (terminated, unterminated): (Vec<usize>, Vec<usize>) = players
        .iter()
        .partition(|player| game.player_outcome(**player).is_some());

    let current_rewards: Vec<f32> = terminated
        .into_iter()
        .map(|player| match game.player_outcome(player).unwrap() {
            Outcome::PlayerWin(bet) => bet,
            Outcome::DealerWin(bet) => -bet,
            Outcome::Push => 0.0,
        })
        .collect();

    let future_rewards: Vec<f32> = unterminated
        .into_iter()
        .map(|_| {
            let state = Model::normalize(game, device);
            let predicted = model.forward(state);
            let predicted_best = predicted
                .max_dim(1)
                .to_data()
                .to_vec::<B::FloatElem>()
                .unwrap()[0]
                .to_f32();

            GAMMA * predicted_best
        })
        .collect();

    [current_rewards, future_rewards].iter().flatten().sum()
}

fn update_targets<B: Backend>(predicted: Tensor<B, 2>, action: usize, reward: f32) -> Tensor<B, 2> {
    let mut data = predicted
        .clone()
        .to_data()
        .to_vec::<B::FloatElem>()
        .unwrap();
    data[action] = B::FloatElem::from_elem(reward);

    Tensor::<B, 2>::from_data(data.as_slice(), &predicted.device())
}
