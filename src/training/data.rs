use burn::data::dataset::Dataset;

use crate::game::{card::Card, game::Game, shoe::Shoe};

pub struct GameDataset {
    count: usize,
}

impl GameDataset {
    pub fn new(count: usize) -> Self {
        Self { count }
    }
}

fn blackjack(cards: [Card; 2]) -> bool {
    cards[0].rank.value() + cards[1].rank.value() == 21
}

impl Dataset<Game> for GameDataset {
    fn get(&self, _ndx: usize) -> Option<Game> {
        let mut shoe = Shoe::new(1);
        shoe.shuffle(&mut rand::rng());

        // re-shuffle games where dealer or player starts with a blackjack
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
        self.count
    }
}
