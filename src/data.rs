use std::{cell::RefCell, rc::Rc, usize};

use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};

use crate::{
    game::{Game, GameStatus, Player},
    model::Model,
    shoe::Shoe,
};

pub struct GameBatcher<B: Backend> {
    device: B::Device,
}

pub struct GameBatch<B: Backend> {
    pub game: Game,
    pub player: Rc<RefCell<Player>>,
    pub input: Tensor<B, 2>,
}

pub struct GameItem {}

pub struct GameDataset {}

impl<B: Backend> GameBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<GameItem, GameBatch<B>> for GameBatcher<B> {
    fn batch(&self, items: Vec<GameItem>) -> GameBatch<B> {
        assert!(items.len() == 1);

        let (game, player) = new_game();
        let input = Model::normalize(&game, &player.borrow(), &self.device);

        GameBatch {
            game,
            player,
            input,
        }
    }
}

impl GameDataset {
    pub fn new() -> Self {
        Self {}
    }

    pub fn len(&self) -> usize {
        1_000
    }
}

impl Dataset<GameItem> for GameDataset {
    fn get(&self, _: usize) -> Option<GameItem> {
        Some(GameItem {})
    }

    fn len(&self) -> usize {
        self.len()
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
