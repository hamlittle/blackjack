use std::path::Path;

use blackjack::{
    game::{game::Game, shoe::Shoe},
    training::{
        data::{GameBatcher, GameDataset},
        model::Action,
        training::TrainingConfig,
    },
};
use burn::{
    backend::{Candle, candle::CandleDevice},
    config::Config,
    data::dataloader::{DataLoaderBuilder, batcher::Batcher},
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{Tensor, cast::ToElement},
};
use strum::IntoEnumIterator;

pub fn main() {
    let device = CandleDevice::default();

    let config = TrainingConfig::load(format!("/tmp/training/config.json"))
        .expect("Config should exist for the model.");

    let record = CompactRecorder::new()
        .load(format!("/tmp/training/model-1.mpk").into(), &device)
        .expect("Trained model should exist.");

    let model = config.model.init::<Candle>(&device).load_record(record);

    let dataset = GameDataset::new(Path::new("out/shoes.1-25000.ndjson"), 10000);
    let batcher = GameBatcher::<Candle>::new(device.clone());
    let loader = DataLoaderBuilder::new(batcher)
        .batch_size(1)
        .num_workers(1)
        .build(dataset);

    let mut action_counts = vec![0; Action::iter().len()];
    for batch in loader.iter() {
        let output = model.forward(batch.input);
        let action = output.argmax(1).flatten::<1>(0, 1).into_scalar().to_usize();

        action_counts[action] += 1;
    }

    println!(
        "Hit: {}, Stand: {}, Double: {}, Surrender: {}",
        action_counts[usize::from(Action::Hit)],
        action_counts[usize::from(Action::Stand)],
        action_counts[usize::from(Action::Double)],
        action_counts[usize::from(Action::Surrender)]
    );

    let test_states = vec![
        // Hard Hands - Hit, Stand, Double
        ([8.0 / 21.0, 5.0 / 11.0, 0.0], "Hard 8 vs. 5"), // Should Hit
        ([9.0 / 21.0, 3.0 / 11.0, 0.0], "Hard 9 vs. 3"), // Should Double
        ([10.0 / 21.0, 6.0 / 11.0, 0.0], "Hard 10 vs. 6"), // Should Double
        ([10.0 / 21.0, 10.0 / 11.0, 0.0], "Hard 10 vs. 10"), // Should Hit
        ([11.0 / 21.0, 8.0 / 11.0, 0.0], "Hard 11 vs. 8"), // Should Double
        ([12.0 / 21.0, 4.0 / 11.0, 0.0], "Hard 12 vs. 4"), // Should Stand
        ([12.0 / 21.0, 10.0 / 11.0, 0.0], "Hard 12 vs. 10"), // Should Hit
        ([14.0 / 21.0, 10.0 / 11.0, 0.0], "Hard 14 vs. 10"), // Should Surrender (if allowed)
        ([15.0 / 21.0, 7.0 / 11.0, 0.0], "Hard 15 vs. 7"), // Should Hit
        ([15.0 / 21.0, 10.0 / 11.0, 0.0], "Hard 15 vs. 10"), // Should Surrender
        ([16.0 / 21.0, 9.0 / 11.0, 0.0], "Hard 16 vs. 9"), // Should Surrender or Stand
        ([17.0 / 21.0, 2.0 / 11.0, 0.0], "Hard 17 vs. 2"), // Should Stand
        ([18.0 / 21.0, 7.0 / 11.0, 0.0], "Hard 18 vs. 7"), // Should Stand
        ([20.0 / 21.0, 6.0 / 11.0, 0.0], "Hard 20 vs. 6"), // Should Stand
        // Soft Hands - Hit, Stand, Double
        ([13.0 / 21.0, 4.0 / 11.0, 1.0], "Soft 13 vs. 4"), // Should Hit
        ([15.0 / 21.0, 6.0 / 11.0, 1.0], "Soft 15 vs. 6"), // Should Double
        ([16.0 / 21.0, 5.0 / 11.0, 1.0], "Soft 16 vs. 5"), // Should Double
        ([17.0 / 21.0, 3.0 / 11.0, 1.0], "Soft 17 vs. 3"), // Should Double
        ([18.0 / 21.0, 4.0 / 11.0, 1.0], "Soft 18 vs. 4"), // Should Double
        ([18.0 / 21.0, 9.0 / 11.0, 1.0], "Soft 18 vs. 9"), // Should Stand or Hit
        ([19.0 / 21.0, 5.0 / 11.0, 1.0], "Soft 19 vs. 5"), // Should Stand
        // Surrender Hands
        ([15.0 / 21.0, 10.0 / 11.0, 0.0], "Hard 15 vs. 10"), // Should Surrender
        ([16.0 / 21.0, 10.0 / 11.0, 0.0], "Hard 16 vs. 10"), // Should Surrender
        ([17.0 / 21.0, 9.0 / 11.0, 0.0], "Hard 17 vs. 9"), // Should Stand, but some surrender strategies exist
        // Borderline Hands
        ([13.0 / 21.0, 2.0 / 11.0, 0.0], "Hard 13 vs. 2"), // Should Stand or Hit
        ([14.0 / 21.0, 2.0 / 11.0, 0.0], "Hard 14 vs. 2"), // Should Stand or Hit
        ([18.0 / 21.0, 9.0 / 11.0, 1.0], "Soft 18 vs. 9"), // Should Stand or Hit
    ];

    for (state_values, desc) in test_states {
        let state = Tensor::<Candle, 2>::from_data([state_values], &device);
        let q_values = model.forward(state.clone());

        println!(
            "{} → Q-values: {:?}",
            desc,
            q_values.to_data().to_vec::<f32>().unwrap()
        );
    }
}
