use blackjack::training::training::TrainingWeightsConfig;

#[rustfmt::skip]
pub const WEIGHTS: [TrainingWeightsConfig; 4] =
[
    TrainingWeightsConfig { learning_rate: 1.0e-3, gamma: 0.90, eps: 1.00, eps_decay: 0.999995, eps_min: 0.80, target_update: 10 },
    TrainingWeightsConfig { learning_rate: 1.0e-3, gamma: 0.95, eps: 0.8, eps_decay: 0.999995, eps_min: 0.50, target_update: 10 },
    TrainingWeightsConfig { learning_rate: 1.0e-3, gamma: 0.99, eps: 0.5, eps_decay: 0.999995, eps_min: 0.20, target_update: 10 },
    TrainingWeightsConfig { learning_rate: 1.0e-3, gamma: 1.00, eps: 0.20, eps_decay: 0.999995, eps_min: 0.05, target_update: 10 },
];
