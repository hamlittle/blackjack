use blackjack::training::training::TrainingWeightsConfig;

#[rustfmt::skip]
pub const WEIGHTS: [TrainingWeightsConfig; 4] =
[
    TrainingWeightsConfig { learning_rate: 1.0e-2, gamma: 0.95, eps_start: 1.0, eps_end: 0.50, offline_update: 5_000 },
    TrainingWeightsConfig { learning_rate: 1.0e-3, gamma: 0.99, eps_start: 1.0, eps_end: 0.50, offline_update: 5_000 },
    TrainingWeightsConfig { learning_rate: 1.0e-3, gamma: 0.99, eps_start: 0.5, eps_end: 0.1, offline_update: 5_000 },
    TrainingWeightsConfig { learning_rate: 1.0e-4, gamma: 1.00, eps_start: 0.1, eps_end: 0.1, offline_update: 5_000 },
];
