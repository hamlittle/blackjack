use std::{path::PathBuf, time::Instant};

use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        TrainingInterrupter,
        metric::{IterationSpeedMetric, LossInput, LossMetric, Metric, MetricMetadata, Numeric},
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
    pub eps: f32,

    #[config(default = 0.0)]
    /// set the exploration decay rate (linear decay each for each training batch)
    pub eps_decay: f32,

    #[config(default = 0.0)]
    /// exploration rate floor
    pub eps_min: f32,
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

        self.config
            .save(format!("{}/config.json", self.config.artifact_dir))
            .expect("Config should be saved successfully");

        // B::seed(self.config.seed);

        info!("Loading datasets...");
        let mut start = Instant::now();

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

        info!("OK! Loaded datasets ({:?}).", start.elapsed());
        info!("Setting up training framework...");
        start = Instant::now();

        let mut learner = Learner::new(
            self.config.model.init::<B>(&device),
            self.config.optimizer.init::<B, Model<B>>(),
            device.clone(),
        );

        info!("OK! set up training framework ({:?}).", start.elapsed());
        info!("Start training...");
        start = Instant::now();

        let interuptor = TrainingInterrupter::new();
        let mut renderer = TuiMetricsRenderer::new(interuptor, None);
        let mut loss_metric = LossMetric::new();
        let mut iteration_metric = IterationSpeedMetric::new();

        for epoch in 0..self.config.num_epochs {
            let mut batch_iter = loader.0.iter();
            let mut iteration = 0;

            while let Some(batch) = batch_iter.next() {
                let weights = self.weights(epoch, iteration);
                let regression = learner.train_step(&batch, &weights);
                learner = learner.optim(&regression, &weights);

                let metadata = MetricMetadata {
                    progress: batch_iter.progress(),
                    epoch: epoch + 1,
                    epoch_total: self.config.num_epochs,
                    iteration: iteration + 1,
                    lr: Some(weights.learning_rate),
                };
                renderer.update_train(MetricState::Numeric(
                    loss_metric.update(&LossInput::<B>::new(regression.loss), &metadata),
                    loss_metric.value(),
                ));
                renderer.update_train(MetricState::Numeric(
                    iteration_metric.update(&(), &metadata),
                    iteration_metric.value(),
                ));
                renderer.render_train(TrainingProgress {
                    progress: batch_iter.progress(),
                    epoch: epoch + 1,
                    epoch_total: self.config.num_epochs,
                    iteration: iteration + 1,
                });

                iteration += 1;
            }

            let mut batch_iter = loader.1.iter();
            let mut iteration = 0;

            while let Some(batch) = batch_iter.next() {
                let weights = self.weights(epoch, iteration * self.config.batch_size);
                let regression = learner.valid_step(&batch, &weights);

                let metadata = MetricMetadata {
                    progress: batch_iter.progress(),
                    epoch: epoch,
                    epoch_total: self.config.num_epochs,
                    iteration,
                    lr: Some(weights.learning_rate),
                };
                renderer.update_valid(MetricState::Numeric(
                    loss_metric.update(&LossInput::<B>::new(regression.loss), &metadata),
                    loss_metric.value(),
                ));
                renderer.update_valid(MetricState::Numeric(
                    iteration_metric.update(&(), &metadata),
                    iteration_metric.value(),
                ));
                renderer.render_valid(TrainingProgress {
                    progress: batch_iter.progress(),
                    epoch,
                    epoch_total: self.config.num_epochs,
                    iteration,
                });

                iteration += 1;
            }

            info!("OK! Finished epoch ({:?})...", start.elapsed());
            start = Instant::now();
        }

        let model_trained = learner.model();
        model_trained
            .save_file(
                format!("{}/model", self.config.artifact_dir),
                &CompactRecorder::new(),
            )
            .expect("Failed to save trained model");

        // let learner = LearnerBuilder::new(artifact_dir)
        //     .metric_train_numeric(LossMetric::new())
        //     .metric_valid_numeric(LossMetric::new())
        //     .with_file_checkpointer(CompactRecorder::new())
        //     .devices(vec![device.clone()])
        //     .num_epochs(config.num_epochs)
        //     .summary()
        //     .build(
        //         config.model.init::<B>(&device),
        //         config.optimizer.init(),
        //         config.learning_rate,
        //     );

        // let model_trained = learner.fit(dataloader_train, dataloader_test);
    }

    fn weights(&self, epoch: usize, iteration: usize) -> Weights {
        let config = if epoch < self.config.weights.len() {
            &self.config.weights[epoch]
        } else {
            &self.config.weights.last().unwrap()
        };

        let progress = iteration * self.config.batch_size;
        let eps = (config.eps * config.eps_decay.powi(progress as i32)).max(config.eps_min);

        Weights {
            learning_rate: config.learning_rate,
            gamma: config.gamma,
            eps,
        }
    }

    // fn forward_step(&self, model: &Model::<B>, )
}
