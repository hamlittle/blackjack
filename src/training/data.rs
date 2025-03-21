use std::collections::VecDeque;

use burn::{
    data::{dataloader::batcher, dataset::Dataset},
    prelude::*,
};
use rand::seq::IteratorRandom;
use rayon::iter::{ParallelBridge, ParallelIterator};

use crate::game::{card::Card, game::Game, shoe::Shoe};

pub struct GameDataset {
    item_count: usize,
    deck_count: usize,
    reshuffle_blackjack: bool,
    shoe_truncate: Option<usize>,
}

impl GameDataset {
    pub fn new(
        item_count: usize,
        deck_count: usize,
        reshuffle_blackjack: bool,
        shoe_truncate: Option<usize>,
    ) -> Self {
        Self {
            item_count,
            deck_count,
            shoe_truncate,
            reshuffle_blackjack,
        }
    }

    fn blackjack(cards: [Card; 2]) -> bool {
        cards[0].rank.value() + cards[1].rank.value() == 21
    }
}

impl Dataset<Game> for GameDataset {
    fn get(&self, _ndx: usize) -> Option<Game> {
        let mut shoe = Shoe::new(self.deck_count);
        shoe.shuffle(&mut rand::rng());

        if self.reshuffle_blackjack {
            while Self::blackjack(shoe.two().unwrap()) || Self::blackjack(shoe.two().unwrap()) {
                shoe.shuffle(&mut rand::rng());
            }
        }

        shoe.reset();
        shoe.truncate(self.shoe_truncate.unwrap_or(self.deck_count * 52));

        // let mut game = Game::new(shoe);
        // game.add_player(1.0);
        // game.start();

        Some(Game::new(shoe))
    }

    fn len(&self) -> usize {
        self.item_count
    }
}

#[derive(Clone, Debug)]
pub struct Batch {
    pub games: Vec<Game>,
}

#[derive(Clone, Debug)]
pub struct Batcher {}

impl Batcher {
    pub fn new() -> Self {
        Self {}
    }
}

impl batcher::Batcher<Game, Batch> for Batcher {
    fn batch(&self, items: Vec<Game>) -> Batch {
        Batch { games: items }
    }
}

#[derive(Clone, Debug)]
pub struct ReplayItem<B>
where
    B: Backend,
{
    pub state: Tensor<B, 2>,
    pub action: Tensor<B, 2, Int>,
    pub reward: Tensor<B, 2>,
    pub terminal: Tensor<B, 2>,
    pub next_state: Tensor<B, 2>,
}

pub struct ReplayBuffer<B>
where
    B: Backend,
{
    buffer: VecDeque<ReplayItem<B>>,
    capacity: usize,
    indices: Vec<Vec<usize>>,
}

impl<B> ReplayBuffer<B>
where
    B: Backend,
{
    pub fn new(capacity: usize, batch_size: usize, iterations: usize) -> Self {
        // pre-compute randomized indices (multithreaded)
        // calling `choose_multiple` is too expensive to keep in the hot-loop
        let indices = (0..iterations)
            .par_bridge()
            .map(|_| (0..capacity).choose_multiple(&mut rand::rng(), batch_size))
            .collect();

        Self {
            buffer: VecDeque::new(),
            capacity,
            indices,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn push(&mut self, item: ReplayItem<B>) {
        if self.buffer.len() > self.capacity {
            self.buffer.pop_front();
        }

        self.buffer.push_back(item);
    }

    pub fn sample(&self, iteration: usize) -> Vec<&ReplayItem<B>> {
        let indices = &self.indices[iteration % self.capacity];
        indices.into_iter().map(|ndx| &self.buffer[*ndx]).collect()
    }
}
