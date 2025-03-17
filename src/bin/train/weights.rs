use blackjack::training::training::TrainingWeightsConfig;

#[rustfmt::skip]
pub const WEIGHTS: [TrainingWeightsConfig; 10] =
[
    // TrainingWeightsConfig { learning_rate: 1.0e-4, gamma: 0.95, eps: 1.00, eps_decay: 0.999995, eps_min: 0.80, target_update: 1 },
    // TrainingWeightsConfig { learning_rate: 1.0e-4, gamma: 0.95, eps: 0.5, eps_decay: 0.999995, eps_min: 0.20, target_update: 1 },
    // TrainingWeightsConfig { learning_rate: 1.0e-4, gamma: 0.99, eps: 0.20, eps_decay: 0.999995, eps_min: 0.05, target_update: 1 },
    // TrainingWeightsConfig { learning_rate: 1.0e-4, gamma: 0.99, eps: 0.05, eps_decay: 0.999995, eps_min: 0.05, target_update: 1 },
    // TrainingWeightsConfig { learning_rate: 5.0e-5, gamma: 0.99, eps: 0.05, eps_decay: 0.999995, eps_min: 0.05, target_update: 1 },
    // TrainingWeightsConfig { learning_rate: 1.0e-6, gamma: 0.99, eps: 0.05, eps_decay: 0.999995, eps_min: 0.05, target_update: 1 },
    // TrainingWeightsConfig { learning_rate: 5.0e-7, gamma: 0.99, eps: 0.05, eps_decay: 0.999995, eps_min: 0.05, target_update: 1 },
    // TrainingWeightsConfig { learning_rate: 1.0e-8, gamma: 0.99, eps: 0.05, eps_decay: 0.999995, eps_min: 0.05, target_update: 1 },
    // TrainingWeightsConfig { learning_rate: 5.0e-9, gamma: 0.99, eps: 0.05, eps_decay: 0.999995, eps_min: 0.05, target_update: 1 },
    // TrainingWeightsConfig { learning_rate: 1.0e-10, gamma: 0.99, eps: 0.05, eps_decay: 0.999995, eps_min: 0.05, target_update: 1 },
    // TrainingWeightsConfig { learning_rate: 5.0e-11, gamma: 0.99, eps: 0.05, eps_decay: 0.999995, eps_min: 0.05, target_update: 1 },
    // TrainingWeightsConfig { learning_rate: 1.0e-12, gamma: 0.99, eps: 0.05, eps_decay: 0.999995, eps_min: 0.05, target_update: 1 },
    TrainingWeightsConfig { learning_rate: 2.0e-3, gamma: 0.95, eps: 0.80, eps_decay: 0.999995, eps_min: 0.50, target_update: 10 },
    TrainingWeightsConfig { learning_rate: 2.0e-3, gamma: 0.95, eps: 0.50, eps_decay: 0.999995, eps_min: 0.05, target_update: 10 },
    TrainingWeightsConfig { learning_rate: 2.0e-3, gamma: 0.99, eps: 0.05, eps_decay: 0.999995, eps_min: 0.00, target_update: 10 },
    TrainingWeightsConfig { learning_rate: 2.0e-3, gamma: 0.99, eps: 0.00, eps_decay: 0.999995, eps_min: 0.00, target_update: 10 },
    TrainingWeightsConfig { learning_rate: 2.0e-3, gamma: 0.99, eps: 0.00, eps_decay: 0.999995, eps_min: 0.00, target_update: 10 },
    TrainingWeightsConfig { learning_rate: 1.0e-4, gamma: 0.99, eps: 0.00, eps_decay: 0.999995, eps_min: 0.00, target_update: 10 },
    TrainingWeightsConfig { learning_rate: 1.0e-4, gamma: 0.99, eps: 0.00, eps_decay: 0.999995, eps_min: 0.00, target_update: 10 },
    TrainingWeightsConfig { learning_rate: 5.0e-5, gamma: 0.99, eps: 0.00, eps_decay: 0.999995, eps_min: 0.00, target_update: 10 },
    TrainingWeightsConfig { learning_rate: 5.0e-5, gamma: 0.99, eps: 0.00, eps_decay: 0.999995, eps_min: 0.00, target_update: 10 },
    TrainingWeightsConfig { learning_rate: 1.0e-6, gamma: 0.99, eps: 0.00, eps_decay: 0.999995, eps_min: 0.00, target_update: 10 },
];
