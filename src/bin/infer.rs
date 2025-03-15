use std::path::Path;

use blackjack::{
    game::{game::Game, shoe::Shoe},
    training::{
        data::{GameBatcher, GameDataset},
        training::TrainingConfig,
    },
};
use burn::{
    config::Config,
    data::dataloader::{DataLoaderBuilder, batcher::Batcher},
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{Tensor, cast::ToElement},
};

pub fn main() {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    let device = WgpuDevice::default();

    let config = TrainingConfig::load(format!("/tmp/training/config.json"))
        .expect("Config should exist for the model.");

    let record = CompactRecorder::new()
        .load(format!("/tmp/training/model").into(), &device)
        .expect("Trained model should exist.");

    let model = config.model.init::<Wgpu>(&device).load_record(record);

    // let dataset = GameDataset::new(Path::new("out/shoes.1-5000.ndjson"), 1000);
    // let batcher = GameBatcher::<Wgpu>::new(device.clone());
    // let loader = DataLoaderBuilder::new(batcher)
    //     .batch_size(1)
    //     .num_workers(1)
    //     .build(dataset);

    // let mut action_counts = vec![0; 2];
    // for batch in loader.iter() {
    //     let output = model.forward(batch.input);
    //     let action = output.argmax(1).flatten::<1>(0, 1).into_scalar().to_usize();

    //     action_counts[action] += 1;
    // }

    // println!("Hit: {}, Stand: {}", action_counts[0], action_counts[1]);

    let test_states = vec![
        // Hard hands
        ([8.0 / 21.0, 6.0 / 11.0, 0.0], "Hard 8 vs. 6"),
        ([10.0 / 21.0, 6.0 / 11.0, 0.0], "Hard 10 vs. 6"),
        ([12.0 / 21.0, 10.0 / 11.0, 0.0], "Hard 12 vs. 10"),
        ([14.0 / 21.0, 7.0 / 11.0, 0.0], "Hard 14 vs. 7"),
        ([16.0 / 21.0, 9.0 / 11.0, 0.0], "Hard 16 vs. 9"),
        ([17.0 / 21.0, 2.0 / 11.0, 0.0], "Hard 17 vs. 2"),
        ([20.0 / 21.0, 5.0 / 11.0, 0.0], "Hard 20 vs. 5"),
        // Soft hands
        ([13.0 / 21.0, 4.0 / 11.0, 1.0], "Soft 13 vs. 4"),
        ([16.0 / 21.0, 6.0 / 11.0, 1.0], "Soft 16 vs. 6"),
        ([18.0 / 21.0, 9.0 / 11.0, 1.0], "Soft 18 vs. 9"),
        ([19.0 / 21.0, 5.0 / 11.0, 1.0], "Soft 19 vs. 5"),
        // Pairs
        ([12.0 / 21.0, 6.0 / 11.0, 0.0], "Pair of 6s vs. 6"),
        ([14.0 / 21.0, 8.0 / 11.0, 0.0], "Pair of 7s vs. 8"),
        ([16.0 / 21.0, 9.0 / 11.0, 0.0], "Pair of 8s vs. 9"),
        ([20.0 / 21.0, 7.0 / 11.0, 0.0], "Pair of 10s vs. 7"),
        ([22.0 / 21.0, 5.0 / 11.0, 0.0], "Pair of Aces vs. 5"),
    ];

    for (state_values, desc) in test_states {
        let state = Tensor::<Wgpu, 2>::from_data([state_values], &device);
        let q_values = model.forward(state.clone());

        println!(
            "{} → Q-values: {:?}",
            desc,
            q_values.to_data().to_vec::<f32>().unwrap()
        );
    }
}
