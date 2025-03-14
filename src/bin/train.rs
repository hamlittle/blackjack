use blackjack::data::{GameBatcher, GameDataset};
use blackjack::model::ModelConfig;
use burn::backend::Autodiff;
use burn::tensor::backend::AutodiffBackend;
use burn::{
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    train::{LearnerBuilder, metric::LossMetric},
};

#[derive(Config)]
pub struct TrainConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,

    #[config(default = 100)]
    pub num_epochs: usize,

    #[config(default = 1)]
    pub batch_size: usize,

    #[config(default = 2)]
    pub num_workers: usize,

    #[config(default = 0x12C0FFEE)]
    pub seed: u64,

    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn run<B: AutodiffBackend>(artifact_dir: &str, config: TrainConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let train_dataset = GameDataset::new();
    let valid_dataset = GameDataset::new();

    println!("Train Dataset Size: {}", train_dataset.len());
    println!("Valid Dataset Size: {}", valid_dataset.len());

    let batcher_train = GameBatcher::<B>::new(device.clone());
    let batcher_test = GameBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(valid_dataset);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Failed to save trained model");
}

fn main() {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    let device = WgpuDevice::default();
    let artifact_dir = "/tmp/blackjack";

    let config = TrainConfig::new(ModelConfig::new(), AdamConfig::new());

    run::<Autodiff<Wgpu>>(artifact_dir, config, device.clone());
}
