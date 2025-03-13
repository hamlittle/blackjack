use blackjack::game::{Game, Outcome, Score};
use blackjack::model::{self, Model};
use burn::backend::{Autodiff, Wgpu, wgpu::WgpuDevice};
use burn::nn::loss::MseLoss;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::tensor::cast::ToElement;

fn simulate_round<B: Backend>(
    model: &mut Model<B>,
    game: &mut Game,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    // find a starting deal that is not an immediate blackjack for either player
    while game.start(1.0) != Outcome::NoWinner {
        game.start(1.0);
    }

    let gamma = 0.99;

    loop {
        let player_score = game.player_score().value();
        let dealer_upcard = game.dealer_upcard().rank.value();
        let is_soft = match game.player_score() {
            Score::Soft(_) => true,
            _ => false,
        };

        let state = Tensor::<B, 2>::from_data(
            [[
                player_score as f32 / 21.0,
                dealer_upcard as f32 / 11.0,
                is_soft as u8 as f32,
            ]],
            device,
        );

        let predicted = model.forward(state.clone());
        let action = predicted
            .clone()
            .argmax(1)
            .to_data()
            .to_vec::<B::IntElem>()
            .unwrap()[0]
            .to_usize();

        let outcome = match action {
            0 => game.hit(),
            1 => game.double(),
            2 => game.stand(),
            _ => panic!("Invalid action"),
        };

        let reward = match outcome {
            Outcome::PlayerWin(r) => r,
            Outcome::DealerWin(r) => -r,
            Outcome::Push => 0.0,
            Outcome::NoWinner => {
                // use predicted value of next state (TD learning)
                let next_player_score = game.player_score().value();
                let next_dealer_upcard = game.dealer_upcard().rank.value();
                let next_is_soft = match game.player_score() {
                    Score::Soft(_) => true,
                    _ => false,
                };

                let next_state = Tensor::<B, 2>::from_data(
                    [[
                        next_player_score as f32 / 21.0,
                        next_dealer_upcard as f32 / 11.0,
                        next_is_soft as u8 as f32,
                    ]],
                    device,
                );

                let next_predicted = model.forward(next_state.clone());

                let max_next_q = next_predicted
                    .clone()
                    .max_dim(1)
                    .to_data()
                    .to_vec::<B::FloatElem>()
                    .unwrap()[0]
                    .to_f32();

                gamma * max_next_q
            }
        };

        let mut target = predicted
            .clone()
            .to_data()
            .to_vec::<B::FloatElem>()
            .unwrap();
        target[action] = B::FloatElem::from_elem(reward);
        let target = Tensor::<B, 2>::from_data(target.as_slice(), device);

        return (state, target);
    }
}

// fn train<B: Backend, O>(
//     epochs: usize,
//     model: &mut model::Model<B>,
//     optimizer: &mut O,
//     loss_fn: &MseLoss,
// ) {
//     for epoch in 0..epochs {
//         let mut total_loss = 0.0;
//         for (state, target_q_values) in &train_data {
//             let predictions = model.forward(state.clone());
//             let loss = loss_fn.forward(predictions, target_q_values.clone());
//             optimizer.step(loss);
//             total_loss += loss.to_data().value()[0];
//         }
//         println!(
//             "Epoch {}: Loss {:.4}",
//             epoch,
//             total_loss / train_data.len() as f32
//         );
//     }
// }

fn main() {
    type Backend = Wgpu<f32, i32>;
    type AutodiffBackend = Autodiff<Backend>;

    let device = WgpuDevice::default();
    let model = model::Config::new(3, 3).init::<Backend>(&device);
    let optimizer = AdamConfig::new().init::<AutodiffBackend, model::Model<AutodiffBackend>>();
    let loss_fn = MseLoss::new();

    println!("{}", model);

    // train(10, &mut model, &mut optimizer, &loss_fn);
}
