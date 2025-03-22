use blackjack::model::dqn::{LearnerConfig, ModelConfig, Weights};
use burn::{
    backend::{Autodiff, Candle, candle::CandleDevice},
    optim::AdamConfig,
};
use train::{Hyper, TrainingConfig};

#[rustfmt::skip]
const WEIGHTS: [Weights; 12] = [
    Weights { lr: 1.0e-4, gamma: 0.990, eps: 0.90, target_update: 5_000 },
    Weights { lr: 5.0e-5, gamma: 0.990, eps: 0.50, target_update: 5_000 },

    Weights { lr: 1.0e-5, gamma: 0.990, eps: 0.30, target_update: 5_000 },
    Weights { lr: 1.0e-5, gamma: 0.990, eps: 0.20, target_update: 5_000 },
    Weights { lr: 5.0e-6, gamma: 0.995, eps: 0.15, target_update: 5_000 },
    Weights { lr: 5.0e-6, gamma: 0.995, eps: 0.10, target_update: 5_000 },

    Weights { lr: 2.5e-6, gamma: 0.995, eps: 0.05, target_update: 10_000 },
    Weights { lr: 2.5e-6, gamma: 0.997, eps: 0.05, target_update: 10_000 },

    Weights { lr: 1.0e-6, gamma: 0.999, eps: 0.02, target_update: 20_000 },
    Weights { lr: 5.0e-7, gamma: 0.999, eps: 0.02, target_update: 20_000 },
    Weights { lr: 2.5e-7, gamma: 0.999, eps: 0.01, target_update: 20_000 },
    Weights { lr: 1.0e-7, gamma: 0.999, eps: 0.01, target_update: 20_000 },
];

#[rustfmt::skip]
const HYPER: [Hyper; 12] = [
    Hyper { weights: WEIGHTS[0],  iterations:   100_000, replay:   5_000,  batch_size: 128, gen_steps: 4, train_steps: 1, trunc: 52 },
    Hyper { weights: WEIGHTS[1],  iterations:   100_000, replay:   5_000,  batch_size: 128, gen_steps: 4, train_steps: 1, trunc: 52 },

    Hyper { weights: WEIGHTS[2],  iterations:   200_000, replay:  25_000,  batch_size: 256, gen_steps: 4, train_steps: 2, trunc: 52 },
    Hyper { weights: WEIGHTS[3],  iterations:   200_000, replay:  25_000,  batch_size: 256, gen_steps: 4, train_steps: 2, trunc: 52 },
    Hyper { weights: WEIGHTS[4],  iterations:   200_000, replay:  25_000,  batch_size: 256, gen_steps: 4, train_steps: 3, trunc: 52 },
    Hyper { weights: WEIGHTS[5],  iterations:   200_000, replay:  25_000,  batch_size: 256, gen_steps: 4, train_steps: 3, trunc: 52 },

    Hyper { weights: WEIGHTS[6],  iterations:   500_000, replay:  50_000,  batch_size: 512, gen_steps: 4, train_steps: 4, trunc: 52 },
    Hyper { weights: WEIGHTS[7],  iterations:   500_000, replay:  50_000,  batch_size: 512, gen_steps: 4, train_steps: 4, trunc: 52 },

    Hyper { weights: WEIGHTS[8],  iterations:   800_000, replay:  150_000, batch_size: 512, gen_steps: 4, train_steps: 6, trunc: 52 },
    Hyper { weights: WEIGHTS[9],  iterations:   800_000, replay:  150_000, batch_size: 512, gen_steps: 4, train_steps: 6, trunc: 52 },
    Hyper { weights: WEIGHTS[10], iterations:   800_000, replay:  150_000, batch_size: 512, gen_steps: 4, train_steps: 8, trunc: 52 },
    Hyper { weights: WEIGHTS[11], iterations:   800_000, replay:  150_000, batch_size: 512, gen_steps: 4, train_steps: 8, trunc: 52 },
];

mod train;

type AutodiffBackend = Autodiff<Candle>;

fn main() {
    env_logger::init();

    let device = CandleDevice::default();

    let model = ModelConfig::new().with_hidden_size(128);
    let optimizer = AdamConfig::new();
    let learner = LearnerConfig::new(model, optimizer);

    let config = TrainingConfig {
        artifact_dir: String::from("/tmp/training"),
        hyper: HYPER.to_vec(),
        learner: learner,
        seed: 0x12C0FFEE,
    };

    let trainer = config.init::<AutodiffBackend>(device);
    trainer.run();
}
