use log::*;
use std::cell::RefCell;

use crate::{
    game::{Game, GameStatus},
    model::Model,
    shoe::Shoe,
};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};

pub struct GameDataset {
    size: usize,
}

impl GameDataset {
    pub fn new(size: usize) -> Self {
        Self { size }
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn new_game(&self) -> Game {
        loop {
            let mut shoe = Shoe::new(1);
            shoe.shuffle(&mut rand::rng());

            let mut game = Game::new(shoe);
            game.add_player(1.0);

            let status = game.start();

            if status == GameStatus::PlayerTurn {
                return game;
            }
        }
    }
}

impl Dataset<Game> for GameDataset {
    fn get(&self, _: usize) -> Option<Game> {
        Some(self.new_game())
    }

    fn len(&self) -> usize {
        self.len()
    }
}

#[derive(Clone, Debug)]
pub struct GameBatcher<B: Backend> {
    device: B::Device,

    exploration: RefCell<f32>,
    decay: f32,
    min: f32,
}

#[derive(Clone, Debug)]
pub struct GameBatch<B: Backend> {
    pub game: Game,
    pub input: Tensor<B, 2>,
    pub exploration: f32,
}

impl<B: Backend> GameBatcher<B> {
    pub fn new(device: B::Device, exploration: f32, decay: f32, min: f32) -> Self {
        Self {
            device,
            exploration: RefCell::new(exploration),
            decay,
            min,
        }
    }
}

impl<B: Backend> Batcher<Game, GameBatch<B>> for GameBatcher<B> {
    fn batch(&self, items: Vec<Game>) -> GameBatch<B> {
        assert!(items.len() == 1);

        let game = items[0].clone();
        let input = Model::normalize(&game, &self.device);

        let mut exploration = *self.exploration.borrow() * self.decay;
        if exploration < self.min {
            exploration = self.min
        }

        *self.exploration.borrow_mut() = exploration;
        // info!("exploration: {}", exploration);

        GameBatch {
            game,
            input,
            exploration,
        }
    }
}
