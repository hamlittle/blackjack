use blackjack::training::training::TrainingWeightsConfig;

#[rustfmt::skip]
pub const WEIGHTS: [TrainingWeightsConfig; 10] =
[
    TrainingWeightsConfig { learning_rate: 3e-3, gamma: 0.99, eps: 1.0, eps_decay: 0.9999, eps_min: 0.1 },
    TrainingWeightsConfig { learning_rate: 2.5e-3, gamma: 0.99, eps: 0.9, eps_decay: 0.9999, eps_min: 0.1 },
    TrainingWeightsConfig { learning_rate: 2e-3, gamma: 0.99, eps: 0.8, eps_decay: 0.9999, eps_min: 0.1 },
    TrainingWeightsConfig { learning_rate: 1.5e-3, gamma: 0.99, eps: 0.7, eps_decay: 0.9999, eps_min: 0.1 },
    TrainingWeightsConfig { learning_rate: 1e-3, gamma: 0.99, eps: 0.6, eps_decay: 0.9999, eps_min: 0.05 },
    TrainingWeightsConfig { learning_rate: 7.5e-4, gamma: 0.99, eps: 0.5, eps_decay: 0.9999, eps_min: 0.05 },
    TrainingWeightsConfig { learning_rate: 5e-4, gamma: 0.99, eps: 0.4, eps_decay: 0.9999, eps_min: 0.05 },
    TrainingWeightsConfig { learning_rate: 3e-4, gamma: 0.99, eps: 0.3, eps_decay: 0.9999, eps_min: 0.05 },
    TrainingWeightsConfig { learning_rate: 2e-4, gamma: 0.99, eps: 0.2, eps_decay: 0.9999, eps_min: 0.05 },
    TrainingWeightsConfig { learning_rate: 1e-4, gamma: 0.99, eps: 0.1, eps_decay: 0.9999, eps_min: 0.05 },
];
