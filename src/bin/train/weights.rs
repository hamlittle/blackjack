use blackjack::training::training::TrainingWeightsConfig;

#[rustfmt::skip]
pub const WEIGHTS: [TrainingWeightsConfig; 10] =
[
        TrainingWeightsConfig { learning_rate: 0.0001, gamma: 0.99, eps: 1.0, eps_decay: 0.999994, eps_min: 0.05 },
        TrainingWeightsConfig { learning_rate: 0.000144, gamma: 0.99, eps: 0.9, eps_decay: 0.999994, eps_min: 0.05 },
        TrainingWeightsConfig { learning_rate: 0.000189, gamma: 0.99, eps: 0.8, eps_decay: 0.999994, eps_min: 0.05 },
        TrainingWeightsConfig { learning_rate: 0.000233, gamma: 0.99, eps: 0.7, eps_decay: 0.999994, eps_min: 0.05 },
        TrainingWeightsConfig { learning_rate: 0.000278, gamma: 0.99, eps: 0.6, eps_decay: 0.999994, eps_min: 0.05 },
        TrainingWeightsConfig { learning_rate: 0.000322, gamma: 0.99, eps: 0.5, eps_decay: 0.999994, eps_min: 0.05 },
        TrainingWeightsConfig { learning_rate: 0.000367, gamma: 0.99, eps: 0.4, eps_decay: 0.999994, eps_min: 0.05 },
        TrainingWeightsConfig { learning_rate: 0.000411, gamma: 0.99, eps: 0.3, eps_decay: 0.999994, eps_min: 0.05 },
        TrainingWeightsConfig { learning_rate: 0.000456, gamma: 0.99, eps: 0.2, eps_decay: 0.999994, eps_min: 0.05 },
        TrainingWeightsConfig { learning_rate: 0.0005, gamma: 0.99, eps: 0.1, eps_decay: 0.999994, eps_min: 0.05 },
    ];
