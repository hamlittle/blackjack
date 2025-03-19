use std::time::Instant;

use blackjack::training::{
    data::GameDataset,
    ppo::{Learner, Model, ModelConfig},
};
use burn::{
    data::dataloader::DataLoaderBuilder, optim::AdamConfig, prelude::*,
    tensor::backend::AutodiffBackend, train::metric::MetricMetadata,
};

use crate::{data::ModelBatcher, metrics::Update, render::Render};

#[derive(Config)]
pub struct Weights {
    pub learning_rate: f32,
    pub gamma: f32,
}

#[derive(Config)]
pub struct TrainingConfig {
    pub artifact_dir: String,
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    pub weights: Vec<Weights>,

    #[config(default = 1_000)]
    pub train_count: usize,

    #[config(default = 1_000)]
    pub valid_count: usize,

    #[config(default = 10)]
    pub num_epochs: usize,

    #[config(default = 32)]
    pub batch_size: usize,

    #[config(default = 0x12C0FFEE)]
    pub seed: u64,

    #[config(default = 1)]
    pub num_workers: usize,
}

impl TrainingConfig {
    pub fn init(self) -> Trainer {
        Trainer::new(self)
    }
}

pub struct Trainer {
    config: TrainingConfig,
}

impl Trainer {
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
    }

    pub fn run<B: AutodiffBackend>(&self, device: B::Device) {
        let mut render = Render::new();

        let mut learner = Learner::new(
            self.config.model.clone().init::<B>(&device),
            self.config.optimizer.clone().init::<B, Model<B>>(),
            device.clone(),
        );

        let loader_train = DataLoaderBuilder::new(ModelBatcher::<B>::new(device.clone()))
            .batch_size(self.config.batch_size)
            .shuffle(self.config.seed)
            .num_workers(self.config.num_workers)
            .build(GameDataset::new(self.config.train_count));

        let loader_valid = DataLoaderBuilder::new(ModelBatcher::<B>::new(device.clone()))
            .batch_size(self.config.batch_size)
            .shuffle(self.config.seed)
            .num_workers(self.config.num_workers)
            .build(GameDataset::new(self.config.valid_count));

        for epoch in 0..self.config.num_epochs {
            let weights = self
                .config
                .weights
                .get(epoch)
                .or(self.config.weights.last())
                .unwrap();

            // train epoch
            render.reset();

            let mut iter = loader_train.iter();
            while let Some(batch) = iter.next() {
                let start = Instant::now();

                let item = learner.train_step(&batch.games, batch.state.clone());
                let loss = item.item.clone();
                learner = learner.fit(item, weights.learning_rate);

                let metadata = MetricMetadata {
                    progress: iter.progress(),
                    epoch: epoch + 1,
                    epoch_total: self.config.num_epochs,
                    iteration: iter.progress().items_processed / self.config.batch_size,
                    lr: Some(weights.learning_rate as f64),
                };
                let update = Update {
                    metadata,
                    win_loss: 0.0,
                    elapsed: start.elapsed(),
                    batch_size: self.config.batch_size,
                    discount: weights.gamma as f64,
                    learning_rate: weights.learning_rate as f64,
                    loss,
                };
                render.update(update, true);
            }

            // valid epoch
            render.reset();

            let mut iter = loader_valid.iter();
            while let Some(batch) = iter.next() {
                let start = Instant::now();

                let item = learner.valid_step(&batch.games, batch.state.clone());
                let loss = item.item.clone();

                let metadata = MetricMetadata {
                    progress: iter.progress(),
                    epoch: epoch + 1,
                    epoch_total: self.config.num_epochs,
                    iteration: iter.progress().items_processed / self.config.batch_size,
                    lr: Some(0.0),
                };
                let update = Update {
                    metadata,
                    win_loss: 0.0,
                    elapsed: start.elapsed(),
                    batch_size: self.config.batch_size,
                    discount: 0.0,
                    learning_rate: 0.0,
                    loss,
                };
                render.update(update, false);
            }
        }
        render.join();
    }
}
