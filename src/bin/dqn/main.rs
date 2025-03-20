use blackjack::model::dqn::{LearnerConfig, ModelConfig, Weights};
use burn::{
    backend::{Autodiff, Candle, candle::CandleDevice},
    optim::AdamConfig,
};
use train::{Hyper, TrainingConfig};

#[rustfmt::skip]
const WEIGHTS: [Weights; 10] = [
    Weights { lr: 1.0e-4, gamma: 0.990, eps: 0.90 },
    Weights { lr: 5.0e-5, gamma: 0.990, eps: 0.90 },
    Weights { lr: 2.5e-5, gamma: 0.990, eps: 0.70 },
    Weights { lr: 1.0e-5, gamma: 0.990, eps: 0.50 },

    Weights { lr: 5.0e-6, gamma: 0.990, eps: 0.30 },
    Weights { lr: 2.5e-6, gamma: 0.990, eps: 0.20 },
    Weights { lr: 1.0e-6, gamma: 0.995, eps: 0.15 },
    Weights { lr: 5.0e-7, gamma: 0.995, eps: 0.10 },

    Weights { lr: 2.5e-8, gamma: 0.995, eps: 0.05 },
    Weights { lr: 1.0e-8, gamma: 0.995, eps: 0.05 },
];

#[rustfmt::skip]
const HYPER: [Hyper; 10] = [
    Hyper { weights: WEIGHTS[0], iterations:    100_000, replay:  25_000, batch_size: 128, gen_steps: 4, train_steps: 1, trunc: 20 },
    Hyper { weights: WEIGHTS[1], iterations:    100_000, replay:  25_000, batch_size: 128, gen_steps: 4, train_steps: 1, trunc: 20 },
    Hyper { weights: WEIGHTS[2], iterations:    100_000, replay:  25_000, batch_size: 128, gen_steps: 4, train_steps: 1, trunc: 20 },
    Hyper { weights: WEIGHTS[3], iterations:    100_000, replay:  25_000, batch_size: 128, gen_steps: 4, train_steps: 1, trunc: 20 },

    Hyper { weights: WEIGHTS[4], iterations:    500_000, replay:  50_000, batch_size: 128, gen_steps: 4, train_steps: 2, trunc: 20 },
    Hyper { weights: WEIGHTS[5], iterations:    500_000, replay:  75_000, batch_size: 256, gen_steps: 4, train_steps: 2, trunc: 20 },
    Hyper { weights: WEIGHTS[6], iterations:    500_000, replay:  75_000, batch_size: 256, gen_steps: 4, train_steps: 2, trunc: 20 },
    Hyper { weights: WEIGHTS[7], iterations:    500_000, replay:  75_000, batch_size: 256, gen_steps: 4, train_steps: 2, trunc: 20 },

    Hyper { weights: WEIGHTS[8], iterations:    4_000_000, replay:  150_000, batch_size: 512, gen_steps: 4, train_steps: 3, trunc: 20 },
    Hyper { weights: WEIGHTS[9], iterations:    4_000_000, replay:  150_000, batch_size: 512, gen_steps: 4, train_steps: 3, trunc: 20 },
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
