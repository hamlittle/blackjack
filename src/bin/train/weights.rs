use blackjack::training::training::TrainingWeightsConfig;

#[rustfmt::skip]
pub const WEIGHTS: [TrainingWeightsConfig; 7] =
[
    TrainingWeightsConfig { learning_rate: 1.0e-8, gamma: 0.99, eps: 1.00, eps_decay: 0.999995, eps_min: 0.05, target_update: 10 },
    TrainingWeightsConfig { learning_rate: 1.0e-8, gamma: 0.99, eps: 0.80, eps_decay: 0.999995, eps_min: 0.05, target_update: 10 },
    TrainingWeightsConfig { learning_rate: 1.0e-8, gamma: 0.99, eps: 0.60, eps_decay: 0.999995, eps_min: 0.05, target_update: 10 },
    TrainingWeightsConfig { learning_rate: 1.0e-8, gamma: 0.99, eps: 0.40, eps_decay: 0.999995, eps_min: 0.05, target_update: 20 },
    TrainingWeightsConfig { learning_rate: 8.0e-9, gamma: 0.99, eps: 0.20, eps_decay: 0.999995, eps_min: 0.05, target_update: 50 },
    TrainingWeightsConfig { learning_rate: 5.0e-9, gamma: 0.99, eps: 0.10, eps_decay: 0.999995, eps_min: 0.05, target_update: 100 },
    TrainingWeightsConfig { learning_rate: 1.0e-9, gamma: 0.99, eps: 0.10, eps_decay: 0.999995, eps_min: 0.05, target_update: 100 },
];
