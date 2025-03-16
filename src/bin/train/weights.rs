use blackjack::training::training::TrainingWeightsConfig;

#[rustfmt::skip]
pub const WEIGHTS: [TrainingWeightsConfig; 8] =
[
    TrainingWeightsConfig { learning_rate: 5.0e-3, gamma: 0.90, eps: 1.00, eps_decay: 0.999995, eps_min: 0.50, target_update: 1 },
    TrainingWeightsConfig { learning_rate: 5.0e-3, gamma: 0.90, eps: 1.00, eps_decay: 0.999995, eps_min: 0.50, target_update: 1 },
    TrainingWeightsConfig { learning_rate: 5.0e-3, gamma: 0.95, eps: 0.8, eps_decay: 0.999995, eps_min: 0.50, target_update: 1 },
    TrainingWeightsConfig { learning_rate: 5.0e-3, gamma: 0.95, eps: 0.8, eps_decay: 0.999995, eps_min: 0.50, target_update: 1 },
    TrainingWeightsConfig { learning_rate: 5.0e-3, gamma: 0.99, eps: 0.5, eps_decay: 0.999995, eps_min: 0.50, target_update: 1 },
    TrainingWeightsConfig { learning_rate: 5.0e-3, gamma: 0.99, eps: 0.5, eps_decay: 0.999995, eps_min: 0.50, target_update: 1 },
    TrainingWeightsConfig { learning_rate: 1.0e-3, gamma: 0.99, eps: 0.10, eps_decay: 0.999995, eps_min: 0.05, target_update: 10 },
    TrainingWeightsConfig { learning_rate: 1.0e-3, gamma: 0.99, eps: 0.10, eps_decay: 0.999995, eps_min: 0.05, target_update: 10 },
];
