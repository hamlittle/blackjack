use blackjack::training::ppo::ModelConfig;
use burn::{
    backend::{Autodiff, Candle, candle::CandleDevice},
    optim::AdamConfig,
};
use train::{TrainingConfig, Weights};

mod data;
mod metrics;
mod render;
mod train;

#[rustfmt::skip]
const WEIGHTS: [Weights; 1] = [
    Weights { learning_rate: 1e-6, gamma: 0.99, },
    // Weights { learning_rate: 1e-8, gamma: 1.0, }
];

type Backend = Candle;
type AutodiffBackend = Autodiff<Backend>;

fn main() {
    env_logger::init();

    let device = CandleDevice::default();

    let config = TrainingConfig::new(
        String::from("/tmp/training"),
        ModelConfig::new().with_hidden_size(128),
        AdamConfig::new(),
        WEIGHTS.to_vec(),
    )
    .with_train_count(50_000_000)
    .with_valid_count(1_000)
    .with_num_epochs(WEIGHTS.len())
    .with_batch_size(32)
    .with_seed(0x12C0FFEE)
    .with_num_workers(1);

    let trainer = config.init();
    trainer.run::<AutodiffBackend>(device.clone());
}
