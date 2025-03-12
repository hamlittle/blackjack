use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use crate::card::{Card, Rank, Suit};

#[derive(Serialize, Deserialize)]
pub struct Shoe {
    cards: Vec<Card>,
}

impl Shoe {
    pub fn new(deck_count: usize) -> Self {
        let mut cards = Vec::new();
        for _ in 0..deck_count {
            for suit in Suit::iter() {
                for rank in Rank::iter() {
                    cards.push(Card::new(rank, suit));
                }
            }
        }
        Shoe { cards }
    }

    pub fn shuffle(&mut self, rng: &mut impl rand::Rng) {
        self.cards.shuffle(rng);
    }
}
