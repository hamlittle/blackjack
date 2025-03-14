use blackjack::{
    model::ModelConfig,
    training::{self, TrainConfig},
};
use burn::{backend::Autodiff, optim::AdamConfig};

fn main() {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    // env_logger::init();

    let device = WgpuDevice::default();
    let artifact_dir = "/tmp/blackjack";

    let model_config = ModelConfig::new();
    let train_config = TrainConfig::new(model_config, AdamConfig::new());

    training::run::<Autodiff<Wgpu>>(artifact_dir, train_config, device.clone());
}
