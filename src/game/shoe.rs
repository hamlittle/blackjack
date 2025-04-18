use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use super::card::{Card, Rank, Suit};

#[derive(Clone, Debug)]
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

        assert_eq!(52 * deck_count, cards.len());

        Shoe { cards, n_dealt: 0 }
    }

    pub fn shuffle(&mut self, rng: &mut impl rand::Rng) {
        self.cards.shuffle(rng);
        self.n_dealt = 0;
    }

    pub fn truncate(&mut self, count: usize) {
        self.cards.truncate(count);
    }

    pub fn one(&mut self) -> Option<Card> {
        let card = self.cards.get(self.n_dealt)?;
        self.n_dealt += 1;

        Some(*card)
    }

    pub fn two(&mut self) -> Option<[Card; 2]> {
        Some([self.one()?, self.one()?])
    }

    pub fn reset(&mut self) {
        self.n_dealt = 0;
    }
}
