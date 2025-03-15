use std::path::PathBuf;

use blackjack::training::{model::ModelConfig, training::TrainingConfig};
use burn::{backend::Autodiff, optim::AdamConfig};

mod weights;

fn main() {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    // env_logger::init();

    let device = WgpuDevice::default();
    let model = ModelConfig::new().with_hidden_size(64);
    let optimizer = AdamConfig::new();
    let trainer = TrainingConfig::new(
        String::from("/tmp/training"),
        model,
        optimizer,
        (PathBuf::from("out/shoes.1-25000.ndmpk"), 25_000),
        (PathBuf::from("out/shoes.1-5000.ndmpk"), 5_000),
        weights::WEIGHTS.into(),
    )
    .with_batch_size(64)
    .with_num_epochs(10);

    let trainer = trainer.init();
    trainer.run::<Autodiff<Wgpu>>(device);
}
