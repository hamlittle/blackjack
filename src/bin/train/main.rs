use std::path::PathBuf;

use blackjack::training::{model::ModelConfig, training::TrainingConfig};
use burn::{
    backend::{Autodiff, Candle, candle::CandleDevice},
    optim::AdamConfig,
};

mod weights;

fn main() {
    let device = CandleDevice::default();
    let model = ModelConfig::new().with_hidden_size(128);
    let optimizer = AdamConfig::new();
    let trainer = TrainingConfig::new(
        String::from("/tmp/training"),
        model,
        optimizer,
        (PathBuf::from("out/shoes.1-25000000.ndjson"), 25_000_000),
        (PathBuf::from("out/shoes.1-1000000.ndjson"), 1_000_000),
        weights::WEIGHTS.into(),
    )
    .with_batch_size(512)
    .with_num_epochs(10)
    .with_num_workers(8);

    let trainer = trainer.init();
    trainer.run::<Autodiff<Candle>>(device);
}
