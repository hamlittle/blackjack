use blackjack::{
    game::{game::Game, shoe::Shoe},
    training::{data::GameBatcher, training::TrainingConfig},
};
use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
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
    // let batcher = GameBatcher::<Wgpu>::new(device.clone(), 0.0, 0.0, 0.0);

    // let mut action_counts = vec![0; 2];
    // for _ in 0..1000 {
    //     let mut shoe = Shoe::new(1);
    //     shoe.shuffle(&mut rand::rng());

    //     let mut game = Game::new(shoe);
    //     game.add_player(1.0);
    //     game.start();

    //     let batch = batcher.batch(vec![game.clone()]);

    //     let output = model.forward(batch.input);
    //     let action = output.argmax(1).flatten::<1>(0, 1).into_scalar().to_usize();

    //     action_counts[action] += 1;
    // }

    // println!("Hit: {}, Stand: {}", action_counts[0], action_counts[1]);

    let test_states = vec![
        ([10.0 / 21.0, 6.0 / 11.0, 0.0], "Hard 10 vs. 6"),
        ([12.0 / 21.0, 10.0 / 11.0, 0.0], "Hard 12 vs. 10"),
        ([17.0 / 21.0, 2.0 / 11.0, 0.0], "Hard 17 vs. 2"),
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
