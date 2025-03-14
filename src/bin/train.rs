use blackjack::data::{GameBatcher, GameDataset};
use blackjack::model::ModelConfig;
use burn::data::dataloader::DataLoaderBuilder;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

#[derive(Config)]
pub struct TrainConfig {
    pub optimizer: AdamConfig,

    #[config(default = 100)]
    pub num_epochs: usize,

    #[config(default = 2)]
    pub num_workers: usize,

    #[config(default = 0xC0FFEE)]
    pub seed: u64,

    #[config(default = 1)]
    pub batch_size: usize,
}

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn run<B: AutodiffBackend>(artifact_dir: &str, device: B::Device) {
    create_artifact_dir(artifact_dir);

    let optimizer = AdamConfig::new();
    let config = TrainConfig::new(optimizer);
    let model = ModelConfig::new().init(&device);
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

    // // Model
    // let learner = LearnerBuilder::new(artifact_dir)
    //     .metric_train_numeric(LossMetric::new())
    //     .metric_valid_numeric(LossMetric::new())
    //     .with_file_checkpointer(CompactRecorder::new())
    //     .devices(vec![device.clone()])
    //     .num_epochs(config.num_epochs)
    //     .summary()
    //     .build(model, config.optimizer.init(), 1e-3);

    // let model_trained = learner.fit(dataloader_train, dataloader_test);

    // config
    //     .save(format!("{artifact_dir}/config.json").as_str())
    //     .unwrap();

    // model_trained
    //     .save_file(
    //         format!("{artifact_dir}/model"),
    //         &NoStdTrainingRecorder::new(),
    //     )
    //     .expect("Failed to save trained model");
}

fn main() {}
