use std::path::PathBuf;

use blackjack::training::{model::ModelConfig, training::TrainingConfig};
use burn::{
    backend::{Autodiff, Candle, candle::CandleDevice},
    optim::AdamConfig,
};
use weights::WEIGHTS;

mod weights;

fn main() {
    env_logger::init();

    let device = CandleDevice::default();
    let model = ModelConfig::new().with_hidden_size(128);
    let optimizer = AdamConfig::new();
    let trainer = TrainingConfig::new(
        String::from("/tmp/training"),
        model,
        optimizer,
        (
            PathBuf::from("out/half-shoe/shoes.1-100000000.ndjson"),
            10_000_000,
        ),
        (
            PathBuf::from("out/half-shoe/shoes.1-1000000.ndjson"),
            5_000_000,
        ),
        weights::WEIGHTS.into(),
    )
    .with_batch_size(1024)
    .with_num_epochs(WEIGHTS.len())
    .with_num_workers(2);

    let trainer = trainer.init();
    trainer.run::<Autodiff<Candle>>(device);
}
