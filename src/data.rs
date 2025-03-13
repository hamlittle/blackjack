use std::{cell::RefCell, rc::Rc};

use burn::{data::dataloader::batcher::Batcher, prelude::*};

use crate::{
    game::{Game, GameStatus, Player, Score},
    shoe::Shoe,
};

pub struct GameBatcher<B: Backend> {
    device: B::Device,
}

pub struct GameBatch<B: Backend> {
    pub game: Game,
    pub player: Rc<RefCell<Player>>,
    pub state: Tensor<B, 2>,
}

impl<B: Backend> GameBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<(), GameBatch<B>> for GameBatcher<B> {
    fn batch(&self, _: Vec<()>) -> GameBatch<B> {
        let (game, player) = new_game();

        let player_score = player.borrow().score();
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
            &self.device,
        );

        GameBatch {
            game,
            player,
            state,
        }
    }
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
