use std::cell::RefCell;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

use blackjack::game::{Game, GameStatus, Outcome, Player, Score};
use blackjack::model::{self, Model};
use blackjack::shoe::Shoe;
use burn::backend::{Autodiff, Wgpu, wgpu::WgpuDevice};
use burn::nn::loss::MseLoss;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::tensor::cast::ToElement;
use num_enum::TryFromPrimitive;

#[derive(TryFromPrimitive)]
#[repr(usize)]
enum Action {
    Hit = 0,
    Stand = 1,
    Double = 2,
    Split = 3,
    Surrender = 4,
}

fn new_game() -> (Game, Rc<RefCell<Player>>) {
    let (game, player) = loop {
        let mut shoe = Shoe::new(1);
        shoe.shuffle(&mut rand::rng());

        let mut game = Game::new(shoe);
        let player = game.add_player(1.0);

        let status = game.start();

        if status == GameStatus::PlayerTurn {
            break (game, player);
        }
    };

    (game, player)
}

fn forward<B: Backend>(
    model: &Model<B>,
    device: &B::Device,
    game: &Game,
    player: &Player,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let player_score = player.score();
    let is_soft = match player_score {
        Score::Soft(_) => true,
        _ => false,
    };
    let dealer_upcard = game.dealer_upcard();

    let state = Tensor::<B, 2>::from_data(
        [[
            player_score.value() as f32 / 21.0,
            is_soft as u8 as f32,
            dealer_upcard.rank.value() as f32 / 11.0,
        ]],
        device,
    );

    (state.clone(), model.forward(state))
}

fn compute_reward<B: Backend>(
    model: &Model<B>,
    device: &B::Device,
    game: &Game,
    plays: &[Rc<RefCell<Player>>],
) -> f32 {
    const GAMMA: f32 = 0.99;

    let reward = plays
        .iter()
        .map(|play| {
            let play = play.borrow();

            if play.outcome().is_some() {
                match play.outcome().unwrap() {
                    Outcome::PlayerWin(bet) => bet,
                    Outcome::DealerWin(bet) => -bet,
                    Outcome::Push => 0.0,
                }
            } else {
                let (_, predicted) = forward(model, device, game, &play);
                let predicted_best = predicted
                    .max_dim(1)
                    .to_data()
                    .to_vec::<B::FloatElem>()
                    .unwrap()[0]
                    .to_f32();

                GAMMA * predicted_best
            }
        })
        .sum();

    reward
}

fn update_target<B: Backend>(predicted: Tensor<B, 2>, action: usize, reward: f32) -> Tensor<B, 2> {
    let mut data = predicted
        .clone()
        .to_data()
        .to_vec::<B::FloatElem>()
        .unwrap();
    data[action] = B::FloatElem::from_elem(reward);

    Tensor::<B, 2>::from_data(data.as_slice(), &predicted.device())
}

fn step<B: Backend>(
    model: &Model<B>,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
    let (mut game, player) = new_game();

    let (state, predicted) = forward(model, device, &game, &player.borrow());

    let action = predicted
        .clone()
        .argmax(1)
        .to_data()
        .to_vec::<B::IntElem>()
        .unwrap()[0]
        .to_usize();

    let plays = match Action::try_from(action).unwrap() {
        Action::Hit => {
            player.borrow_mut().hit();
            vec![player]
        }
        Action::Stand => {
            player.borrow_mut().stand();
            vec![player]
        }
        Action::Double => {
            player.borrow_mut().double();
            vec![player]
        }
        Action::Split => {
            let second_play = game.split_player(&mut *player.borrow_mut());
            vec![player, second_play]
        }
        Action::Surrender => {
            player.borrow_mut().surrender();
            vec![player]
        }
    };

    let reward = compute_reward(model, device, &game, &plays);
    let target = update_target(predicted.clone(), action, reward);

    (state, predicted, target)
}

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
