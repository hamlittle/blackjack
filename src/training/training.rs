use std::{
    path::PathBuf,
    sync::mpsc,
    thread::{self, sleep},
    time::{Duration, Instant},
};

use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    optim::AdamConfig,
    prelude::Backend,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        RegressionOutput, TrainingInterrupter,
        metric::{
            IterationSpeedMetric, LossInput, LossMetric, Metric, MetricEntry, MetricMetadata,
            Numeric,
            state::{FormatOptions, NumericMetricState},
        },
        renderer::{MetricState, MetricsRenderer, TrainingProgress, tui::TuiMetricsRenderer},
    },
};
use log::info;

use crate::training::{learner::Learner, model::Model};

use super::{
    data::{GameBatcher, GameDataset},
    learner::Weights,
    model::ModelConfig,
};

#[derive(Config)]
pub struct TrainingWeightsConfig {
    #[config(default = 1e-4)]
    pub learning_rate: f64,

    #[config(default = 0.99)]
    /// discount future rewards
    pub gamma: f32,

    #[config(default = 0.0)]
    /// set the exploration rate
    pub eps: f64,

    #[config(default = 0.0)]
    /// set the exploration decay rate (linear decay each for each training batch)
    pub eps_decay: f64,

    #[config(default = 0.0)]
    /// exploration rate floor
    pub eps_min: f64,

    #[config(default = 100)]
    /// exploration rate floor
    pub target_update: u32,
}

#[derive(Config)]
pub struct TrainingConfig {
    pub artifact_dir: String,
    pub model: ModelConfig,
    pub optimizer: AdamConfig,

    /// data (source, size) for training
    pub train: (PathBuf, usize),
    /// data (source, size) for validation
    pub valid: (PathBuf, usize),

    /// training weights are selected by the current epoch, or the last if epochs > len(weights)
    pub weights: Vec<TrainingWeightsConfig>,

    #[config(default = 10)]
    pub num_epochs: usize,

    #[config(default = 32)]
    pub batch_size: usize,

    #[config(default = 0x12C0FFEE)]
    pub seed: u64,

    #[config(default = 1)]
    pub num_workers: usize,
}

struct Metrics<B: Backend> {
    pub loss: LossMetric<B>,
    pub iteration: IterationSpeedMetric,
    pub batch: NumericMetricState,
    pub exploration: NumericMetricState,
    pub discont: NumericMetricState,
    pub learning_rate: NumericMetricState,
    pub updates: Vec<MetricEntry>,
}

impl<B: Backend> Metrics<B> {
    pub fn new() -> Self {
        Self {
            loss: LossMetric::new(),
            iteration: IterationSpeedMetric::new(),
            batch: NumericMetricState::new(),
            exploration: NumericMetricState::new(),
            discont: NumericMetricState::new(),
            learning_rate: NumericMetricState::new(),
            updates: Vec::new(),
        }
    }

    pub fn update(
        &mut self,
        metadata: &MetricMetadata,
        elapsed: Duration,
        config: &TrainingConfig,
        weights: &Weights,
        regression: &RegressionOutput<B>,
    ) -> Vec<(MetricEntry, f64)> {
        let mut updates = Vec::new();

        updates.push((
            self.loss
                .update(&LossInput::<B>::new(regression.loss.clone()), &metadata),
            self.loss.value(),
        ));
        updates.push((
            self.iteration.update(&(), &metadata),
            self.iteration.value(),
        ));
        updates.push((
            self.batch.update(
                config.batch_size as f64 / elapsed.as_secs_f64(),
                config.batch_size,
                FormatOptions::new("batch").unit("item/second").precision(0),
            ),
            self.batch.value(),
        ));
        updates.push((
            self.exploration.update(
                weights.eps as f64,
                config.batch_size,
                FormatOptions::new("exploration").precision(2),
            ),
            self.exploration.value(),
        ));
        updates.push((
            self.discont.update(
                weights.gamma as f64,
                config.batch_size,
                FormatOptions::new("discount").precision(2),
            ),
            self.discont.value(),
        ));
        updates.push((
            self.learning_rate.update(
                weights.learning_rate as f64,
                config.batch_size,
                FormatOptions::new("learning rate").precision(2),
            ),
            self.learning_rate.value(),
        ));

        updates
    }
}

impl TrainingConfig {
    pub fn init(&self) -> Trainer {
        Trainer::new(self)
    }
}

pub struct Trainer {
    config: TrainingConfig,
}

impl Trainer {
    pub fn new(config: &TrainingConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    pub fn run<B: AutodiffBackend>(&self, device: B::Device) {
        std::fs::remove_dir_all(&self.config.artifact_dir).ok();
        std::fs::create_dir_all(&self.config.artifact_dir).ok();

        let interuptor = TrainingInterrupter::new();
        let mut renderer = TuiMetricsRenderer::new(interuptor, None);

        let (tx, rx) = mpsc::channel();
        let render_thread = thread::spawn(move || {
            rx.iter().for_each(
                |(ndx, metadata, updates): (usize, MetricMetadata, Vec<(MetricEntry, f64)>)| {
                    match ndx {
                        0 => {
                            for update in updates {
                                renderer.update_train(MetricState::Numeric(update.0, update.1));
                            }

                            renderer.render_train(TrainingProgress {
                                progress: metadata.progress,
                                epoch: metadata.epoch,
                                epoch_total: metadata.epoch_total,
                                iteration: metadata.iteration,
                            });
                        }
                        1 => {
                            for update in updates {
                                renderer.update_valid(MetricState::Numeric(update.0, update.1));
                            }

                            renderer.render_valid(TrainingProgress {
                                progress: metadata.progress,
                                epoch: metadata.epoch,
                                epoch_total: metadata.epoch_total,
                                iteration: metadata.iteration,
                            });
                        }
                        _ => panic!("Invalid ndx."),
                    }
                },
            );

            renderer.persistent();
        });

        self.config
            .save(format!("{}/config.json", self.config.artifact_dir))
            .expect("Config should be saved successfully");

        // B::seed(self.config.seed);

        info!("Loading datasets...");

        let dataset = (
            GameDataset::new(&self.config.train.0, self.config.train.1),
            GameDataset::new(&self.config.valid.0, self.config.valid.1),
        );

        let batcher = (
            GameBatcher::<B>::new(device.clone()),
            GameBatcher::<B>::new(device.clone()),
        );

        let loader = (
            DataLoaderBuilder::new(batcher.0)
                .batch_size(self.config.batch_size)
                .shuffle(self.config.seed)
                .num_workers(self.config.num_workers)
                .build(dataset.0),
            DataLoaderBuilder::new(batcher.1)
                .batch_size(self.config.batch_size)
                .shuffle(self.config.seed)
                .num_workers(self.config.num_workers)
                .build(dataset.1),
        );

        info!("OK! Loaded datasets.");
        info!("Setting up training framework...");

        let mut learner = Learner::new(
            self.config.model.init::<B>(&device),
            self.config.optimizer.init::<B, Model<B>>(),
            device.clone(),
        );

        info!("OK! set up training framework.");
        info!("Start training...");

        for epoch in 0..self.config.num_epochs {
            let mut metrics = (Metrics::<B>::new(), Metrics::<B>::new());

            let mut batch_iter = loader.0.iter();
            let mut iteration = 0;

            let mut start = Instant::now();
            while let Some(batch) = batch_iter.next() {
                let weights = self.weights(epoch, iteration);
                let regression = learner.train_step(&batch, &weights);
                learner = learner.optim(&regression, &weights, iteration);

                let metadata = MetricMetadata {
                    progress: batch_iter.progress(),
                    epoch: epoch + 1,
                    epoch_total: self.config.num_epochs,
                    iteration: iteration + 1,
                    lr: Some(weights.learning_rate),
                };
                let updates = metrics.0.update(
                    &metadata,
                    start.elapsed(),
                    &self.config,
                    &weights,
                    &regression,
                );

                let _ = tx.send((0, metadata, updates));

                iteration += 1;
                start = Instant::now();
            }

            let mut batch_iter = loader.1.iter();
            let mut iteration = 0;

            let mut start = Instant::now();
            while let Some(batch) = batch_iter.next() {
                let weights = self.weights(epoch, iteration);
                let regression = learner.valid_step(&batch, &weights);

                let metadata = MetricMetadata {
                    progress: batch_iter.progress(),
                    epoch: epoch + 1,
                    epoch_total: self.config.num_epochs,
                    iteration: iteration + 1,
                    lr: Some(weights.learning_rate),
                };
                let updates = metrics.1.update(
                    &metadata,
                    start.elapsed(),
                    &self.config,
                    &weights,
                    &regression,
                );

                let _ = tx.send((1, metadata, updates));

                iteration += 1;
                start = Instant::now();
            }

            let model_trained = learner.model();
            model_trained
                .save_file(
                    format!("{}/model-{}.", self.config.artifact_dir, epoch),
                    &CompactRecorder::new(),
                )
                .expect("Failed to save trained model");
        }

        info!("OK training completed.");

        let model_trained = learner.model();
        model_trained
            .save_file(
                format!("{}/model", self.config.artifact_dir),
                &CompactRecorder::new(),
            )
            .expect("Failed to save trained model");

        info!("Saved model to {}.", self.config.artifact_dir);

        drop(tx);
        let _ = render_thread.join();
    }

    fn weights(&self, epoch: usize, iteration: usize) -> Weights {
        let config = if epoch < self.config.weights.len() {
            &self.config.weights[epoch]
        } else {
            &self.config.weights.last().unwrap()
        };

        let eps = (config.eps * config.eps_decay.powi(iteration as i32)).max(config.eps_min);

        Weights {
            learning_rate: config.learning_rate,
            gamma: config.gamma,
            eps,
            target_update: config.target_update,
        }
    }
}
