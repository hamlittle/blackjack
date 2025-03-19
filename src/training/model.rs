use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use strum::{EnumIter, IntoEnumIterator};

use crate::game::{card::Card, game::Score};

use super::simulation::{Action, Simulation};

#[derive(IntoPrimitive, EnumIter)]
#[repr(usize)]
enum Input {
    PlayerScore,
    IsSoft,
    DealerUpCard,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = 128)]
    hidden_size: usize,

    #[config(default = 0.01)]
    activation_slope: f64,
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    // fc3: Linear<B>,
    activation: Relu,
    // activation: LeakyRelu,
    // advantage: Linear<B>,
    value: Linear<B>,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model::new(self, device)
    }
}

impl<B: Backend> Model<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        let fc1 = LinearConfig::new(Model::<B>::input_size(), config.hidden_size).init(device);
        let fc2 = LinearConfig::new(config.hidden_size, config.hidden_size).init(device);
        // let fc3 = LinearConfig::new(config.hidden_size, config.hidden_size).init(device);
        // let activation = LeakyReluConfig::new()
        //     .with_negative_slope(config.activation_slope)
        //     .init();
        let activation = Relu::new();
        // let advantage =
        //     LinearConfig::new(config.hidden_size, Model::<B>::output_size()).init(device);
        // let value = LinearConfig::new(config.hidden_size, 1).init(device);
        let value = LinearConfig::new(config.hidden_size, Self::output_size()).init(device);

        Self {
            fc1,
            fc2,
            // fc3,
            activation,
            // advantage,
            value,
        }
    }

    pub fn input_size() -> usize {
        Input::iter().len()
    }

    pub fn output_size() -> usize {
        // Model does not evaluate splits
        Action::iter().len() - 1
    }

    pub fn normalize(player_score: Score, dealer_upcard: Card, device: &B::Device) -> Tensor<B, 2> {
        let data: Vec<B::FloatElem> = Input::iter()
            .map(|input| match input {
                Input::PlayerScore => player_score.value() as f32 / 21.0,
                Input::IsSoft => match player_score {
                    Score::Soft(_) => true as u8 as f32,
                    Score::Hard(_) => false as u8 as f32,
                },
                Input::DealerUpCard => dealer_upcard.rank.value() as f32 / 11.0 as f32,
            })
            .map(|elem| B::FloatElem::from_elem(elem))
            .collect();

        let data = TensorData::new(data, [1, Self::input_size()]);

        Tensor::from_data(data, device)
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(input);
        let x = self.activation.forward(x);
        let x = self.fc2.forward(x);
        let x = self.activation.forward(x);
        // let x = self.fc3.forward(x);
        // let x = self.activation.forward(x);
        // let advantage = self.advantage.forward(x.clone());
        let value = self.value.forward(x);

        // value + (advantage.clone() - advantage.mean_dim(1))
        value
    }
}

// impl<B: AutodiffBackend> TrainStep<GameBatch<B>, RegressionOutput<B>> for Model<B> {
//     fn step(&self, item: GameBatch<B>) -> TrainOutput<RegressionOutput<B>> {
//         let item = self.forward_step(item);
//         let gradients = item.loss.backward();

//         TrainOutput::new(self, gradients, item)
//     }
// }

// impl<B: Backend> ValidStep<GameBatch<B>, RegressionOutput<B>> for Model<B> {
//     fn step(&self, item: GameBatch<B>) -> RegressionOutput<B> {
//         self.forward_step(item)
//     }
// }
