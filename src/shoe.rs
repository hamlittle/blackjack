use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use crate::card::{Card, Rank, Suit};

pub struct Shoe {
    cards: Vec<Card>,
    n_dealt: usize,
}

impl Serialize for Shoe {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.cards.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Shoe {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let cards = Vec::<Card>::deserialize(deserializer)?;
        Ok(Shoe { cards, n_dealt: 0 })
    }
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
        Shoe { cards, n_dealt: 0 }
    }

    pub fn shuffle(&mut self, rng: &mut impl rand::Rng) {
        self.cards.shuffle(rng);
    }

    pub fn next(&mut self) -> Option<&Card> {
        let card = self.cards.get(self.n_dealt);

        if card.is_some() {
            self.n_dealt += 1;
        }

        card
    }
}
