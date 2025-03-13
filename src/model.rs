use burn::{
    nn::{Linear, LinearConfig},
    prelude::*,
    tensor::activation::relu,
};

#[derive(Config, Debug)]
pub struct Config {
    num_inputs: usize,
    num_actions: usize,
    #[config(default = 64)]
    hidden_size: usize,
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    advantage: Linear<B>,
    value: Linear<B>,
}

impl Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model::new(self, device)
    }
}

impl<B: Backend> Model<B> {
    pub fn new(config: &Config, device: &B::Device) -> Self {
        let fc1 = LinearConfig::new(config.num_inputs, config.hidden_size).init(device);
        let fc2 = LinearConfig::new(config.hidden_size, config.hidden_size).init(device);
        let advantage = LinearConfig::new(config.hidden_size, config.num_actions).init(device);
        let value = LinearConfig::new(config.hidden_size, 1).init(device);

        Self {
            fc1,
            fc2,
            advantage,
            value,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(input);
        let x = relu(x);
        let x = self.fc2.forward(x);
        let x = relu(x);
        let advantage = self.advantage.forward(x.clone());
        let value = self.value.forward(x);

        value + (advantage.clone() - advantage.mean_dim(1))
    }
}
