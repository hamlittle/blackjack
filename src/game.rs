use crate::{
    card::{Card, Rank},
    shoe::Shoe,
};
pub struct Game {
    shoe: Shoe,
    player_hand: Vec<Card>,
    dealer_hand: Vec<Card>,
    bet: f32,
}

pub enum Score {
    Bust(u8),
    Hard(u8),
    Soft(u8),
}

impl Score {
    pub fn value(&self) -> u8 {
        match self {
            Score::Bust(score) => *score,
            Score::Hard(score) => *score,
            Score::Soft(score) => *score,
        }
    }
}

#[derive(PartialEq)]
pub enum Outcome {
    PlayerWin(f32),
    DealerWin(f32),
    Push,
    NoWinner,
}

impl Game {
    pub fn new(shoe: Shoe) -> Self {
        Self {
            shoe,
            player_hand: Vec::new(),
            dealer_hand: Vec::new(),
            bet: 0.0,
        }
    }

    pub fn start(&mut self, bet: f32) -> Outcome {
        self.bet = bet;
        self.player_hand.clear();
        self.dealer_hand.clear();

        self.player_hand.push(*self.shoe.next().unwrap());
        self.dealer_hand.push(*self.shoe.next().unwrap());
        self.player_hand.push(*self.shoe.next().unwrap());
        self.dealer_hand.push(*self.shoe.next().unwrap());

        let dealer_blackjack = self.dealer_score().value() == 21;
        let player_blackjack = self.player_score().value() == 21;

        if dealer_blackjack && player_blackjack {
            Outcome::Push
        } else if dealer_blackjack {
            Outcome::DealerWin(self.bet)
        } else if player_blackjack {
            Outcome::PlayerWin(self.bet * 1.5)
        } else {
            Outcome::NoWinner
        }
    }

    pub fn hit(&mut self) -> Outcome {
        self.player_hand.push(*self.shoe.next().unwrap());

        match self.player_score() {
            Score::Bust(_) => Outcome::DealerWin(self.bet),
            _ => Outcome::NoWinner,
        }
    }

    pub fn double(&mut self) -> Outcome {
        self.bet *= 2.0;
        self.hit()
    }

    pub fn stand(&mut self) -> Outcome {
        match self.player_score() {
            Score::Bust(_) => return Outcome::DealerWin(self.bet),
            _ => (),
        }

        while dealer_must_hit(self.dealer_score()) {
            self.dealer_hand.push(*self.shoe.next().unwrap());
        }

        match self.dealer_score() {
            Score::Bust(_) => return Outcome::PlayerWin(self.bet),
            _ => (),
        }

        match self
            .player_score()
            .value()
            .cmp(&self.dealer_score().value())
        {
            std::cmp::Ordering::Less => Outcome::DealerWin(self.bet),
            std::cmp::Ordering::Equal => Outcome::Push,
            std::cmp::Ordering::Greater => Outcome::PlayerWin(self.bet),
        }
    }

    pub fn player_score(&self) -> Score {
        score(&self.player_hand)
    }

    pub fn dealer_upcard(&self) -> &Card {
        &self.dealer_hand[0]
    }

    fn dealer_score(&self) -> Score {
        score(&self.dealer_hand)
    }
}

fn score(hand: &[Card]) -> Score {
    let mut score = hand.iter().map(|card| card.rank.value()).sum();
    let mut aces = hand.iter().filter(|card| card.rank == Rank::Ace).count();

    while score > 21 && aces > 0 {
        aces -= 1;
        score -= 10;
    }

    if aces > 0 {
        Score::Soft(score)
    } else if score <= 21 {
        Score::Hard(score)
    } else {
        Score::Bust(score)
    }
}

fn dealer_must_hit(score: Score) -> bool {
    match score {
        Score::Bust(_) => false,
        Score::Hard(score) => score < 17,
        Score::Soft(score) => score <= 17,
    }
}
