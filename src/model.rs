use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
};

#[derive(Config, Debug)]
pub struct ModelConfig {
    num_inputs: usize,
    num_actions: usize,
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
        let fc1 = LinearConfig::new(config.num_inputs, config.hidden_size).init(device);
        let fc2 = LinearConfig::new(config.hidden_size, config.hidden_size).init(device);
        let activation = Relu::new();
        let advantage = LinearConfig::new(config.hidden_size, config.num_actions).init(device);
        let value = LinearConfig::new(config.hidden_size, 1).init(device);

        Self {
            fc1,
            fc2,
            activation,
            advantage,
            value,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(input);
        let x = self.fc2.forward(x);
        let x = self.activation.forward(x);
        let advantage = self.advantage.forward(x.clone());
        let value = self.value.forward(x);

        value + (advantage.clone() - advantage.mean_dim(1))
    }
}
