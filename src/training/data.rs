use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};
// use log::info;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{
    // fs::File,
    // io::{BufRead, BufReader},
    path::Path,
    // time::Instant,
};

use crate::{
    game::{card::Card, game::Game, shoe::Shoe},
    training::model::Model,
};

pub struct GameDataset {
    // games: Vec<Game>,
    /// loops if size is more than number of games
    size: usize,
}

impl GameDataset {
    pub fn new(_source: &Path, size: usize) -> Self {
        // let start = Instant::now();

        // let reader = BufReader::new(File::open(source).unwrap());
        // let games: Vec<_> = reader
        //     .lines()
        //     .par_bridge()
        //     .map(|line| {
        //         let shoe: Shoe = serde_json::from_str(&line.unwrap()).unwrap();
        //         let mut game = Game::new(shoe);
        //         game.add_player(1.0);
        //         game.start();

        //         game
        //     })
        //     .collect();

        // let duration = start.elapsed();
        // info!(
        //     "OK! Read {} shoes from '{:?}' ({:.3?})",
        //     games.len(),
        //     source,
        //     duration
        // );

        // Self { size, games }
        Self { size }
    }
}

fn blackjack(cards: [Card; 2]) -> bool {
    cards[0].rank.value() + cards[1].rank.value() == 21
}

impl Dataset<Game> for GameDataset {
    fn get(&self, _ndx: usize) -> Option<Game> {
        // // loop if ndx is more than number of games stored
        // let ndx = ndx % self.games.len();

        // Some(self.games[ndx].clone())

        let mut shoe = Shoe::new(1);
        shoe.shuffle(&mut rand::rng());
        shoe.truncate(20);

        while blackjack(shoe.two().unwrap()) || blackjack(shoe.two().unwrap()) {
            shoe.shuffle(&mut rand::rng());
        }
        shoe.reset();

        let mut game = Game::new(shoe);
        game.add_player(1.0);
        game.start();

        Some(game)
    }

    fn len(&self) -> usize {
        self.size
    }
}

#[derive(Clone, Debug)]
pub struct GameBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct GameBatch<B: Backend> {
    pub games: Vec<Game>,
    pub input: Tensor<B, 2>,
}

impl<B: Backend> GameBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<Game, GameBatch<B>> for GameBatcher<B> {
    fn batch(&self, items: Vec<Game>) -> GameBatch<B> {
        let input: Vec<_> = items
            .par_iter()
            .map(|game| Model::normalize(game.player_score(0), game.dealer_upcard(), &self.device))
            .collect();
        let input = Tensor::cat(input, 0);

        assert_eq!([items.len(), Model::<B>::input_size()], input.dims());

        GameBatch {
            games: items,
            input,
        }
    }
}
