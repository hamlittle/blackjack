use std::{cell::RefCell, rc::Rc};

use crate::{
    card::{Card, Rank},
    shoe::Shoe,
};

#[derive(Copy, Clone, PartialEq)]
pub enum GameStatus {
    PlaceBets,
    PlayerTurn,
    GameOver,
}

pub struct Game {
    shoe: Rc<RefCell<Shoe>>,
    dealer_hand: Vec<Card>,
    players: Vec<Rc<RefCell<Player>>>,
    status: GameStatus,
}

pub struct Player {
    shoe: Rc<RefCell<Shoe>>,
    hand: Vec<Card>,
    bet: f32,
    stand: bool,
    outcome: Option<Outcome>,
}

#[derive(Copy, Clone)]
pub enum Outcome {
    PlayerWin(f32),
    DealerWin(f32),
    Push,
}

pub enum Score {
    Hard(u8),
    Soft(u8),
}

impl Player {
    pub fn new(shoe: Rc<RefCell<Shoe>>, bet: f32) -> Self {
        Self {
            shoe: shoe.clone(),
            hand: Vec::new(),
            bet,
            stand: false,
            outcome: None,
        }
    }

    pub fn score(&self) -> Score {
        score(&self.hand)
    }

    pub fn outcome(&self) -> Option<Outcome> {
        self.outcome
    }

    pub fn hit(&mut self) {
        if self.stand {
            panic!("Cannot hit after standing.");
        }

        self.deal_one();

        if self.score().bust() {
            self.outcome = Some(Outcome::DealerWin(self.bet));
            self.stand = true;
        }
    }

    pub fn stand(&mut self) {
        self.stand = true;
    }

    pub fn double(&mut self) {
        if self.stand {
            panic!("Cannot double after standing.");
        }

        if self.hand.len() != 2 {
            panic!("Cannot double after taking an action.");
        }

        self.bet *= 2.0;
        self.hit();
        self.stand();
    }

    fn split_hand(&mut self) -> Card {
        if self.stand {
            panic!("Cannot split after standing.");
        }

        if self.hand.len() != 2 {
            panic!("Cannot split after taking an action.");
        }

        if self.hand[0].rank != self.hand[1].rank {
            panic!("Cannot split unless both cards have the same rank.");
        }

        self.hand.pop().unwrap()
    }

    pub fn surrender(&mut self) {
        if self.stand {
            panic!("Cannot surrender after standing.");
        }

        if self.hand.len() != 2 {
            panic!("Cannot surrender after taking an action.");
        }

        self.outcome = Some(Outcome::DealerWin(self.bet / 2.0));
        self.stand = true;
    }

    fn deal_one(&mut self) {
        self.hand.push(self.shoe.borrow_mut().one().unwrap());
    }

    fn dealer_blackjack(&mut self) {
        if self.score().blackjack() {
            self.outcome = Some(Outcome::Push);
        } else {
            self.outcome = Some(Outcome::DealerWin(self.bet));
        }

        self.stand = true;
    }

    fn player_blackjack(&mut self) {
        self.outcome = Some(Outcome::PlayerWin(self.bet * 1.5));
        self.stand = true;
    }

    fn compare(&mut self, dealer_score: &Score) {
        if !self.stand {
            panic!("Cannot compare until player stands.");
        }

        if self.outcome.is_some() {
            return;
        }

        if dealer_score.bust() {
            self.outcome = Some(Outcome::PlayerWin(self.bet));
            return;
        }

        match self.score().value().cmp(&dealer_score.value()) {
            std::cmp::Ordering::Less => self.outcome = Some(Outcome::DealerWin(self.bet)),
            std::cmp::Ordering::Equal => self.outcome = Some(Outcome::Push),
            std::cmp::Ordering::Greater => self.outcome = Some(Outcome::PlayerWin(self.bet)),
        }
    }
}

impl Game {
    pub fn new(shoe: Shoe) -> Self {
        let shoe = Rc::new(RefCell::new(shoe));

        Self {
            shoe,
            dealer_hand: Vec::new(),
            players: Vec::new(),
            status: GameStatus::PlaceBets,
        }
    }

    pub fn add_player(&mut self, bet: f32) -> Rc<RefCell<Player>> {
        if self.status != GameStatus::PlaceBets {
            panic!("Cannot add player after game has started.");
        }

        let player = Rc::new(RefCell::new(Player::new(self.shoe.clone(), bet)));
        self.players.push(player.clone());

        player
    }

    pub fn split_player(&mut self, player: &mut Player) -> Rc<RefCell<Player>> {
        if self.status != GameStatus::PlayerTurn {
            panic!("Cannot split unless player turn.");
        }

        let card = player.split_hand();

        let mut second_play = Player::new(self.shoe.clone(), player.bet);
        second_play.hand.push(card);

        player.deal_one();
        second_play.deal_one();

        let second_play = Rc::new(RefCell::new(second_play));
        self.players.push(second_play.clone());

        second_play
    }

    pub fn start(&mut self) -> GameStatus {
        if self.status != GameStatus::PlaceBets {
            panic!("Game has already started.");
        }

        for _ in 0..2 {
            self.players.iter_mut().for_each(|player| {
                player.borrow_mut().deal_one();
            });

            self.deal_one();
        }

        if self.dealer_score().blackjack() {
            self.players.iter_mut().for_each(|player| {
                player.borrow_mut().dealer_blackjack();
            });

            self.status = GameStatus::GameOver;
        } else {
            self.players.iter_mut().for_each(|player| {
                if player.borrow().score().blackjack() {
                    player.borrow_mut().player_blackjack();
                }
            });

            if self.players.iter().all(|player| player.borrow().stand) {
                self.status = GameStatus::GameOver;
            } else {
                self.status = GameStatus::PlayerTurn;
            }
        }

        self.status
    }

    pub fn end(&mut self) {
        if self.status != GameStatus::PlayerTurn {
            panic!("Game is not in progress.");
        }

        if !self.players.iter().all(|player| player.borrow().stand) {
            panic!("Cannot end game until all players stand.");
        }

        while dealer_must_hit(self.dealer_score()) {
            self.deal_one();
        }

        let dealer_score = self.dealer_score();

        self.players.iter_mut().for_each(|player| {
            player.borrow_mut().compare(&dealer_score);
        });

        self.status = GameStatus::GameOver;
    }

    pub fn dealer_score(&self) -> Score {
        score(&self.dealer_hand)
    }

    pub fn dealer_upcard(&self) -> Card {
        self.dealer_hand[0]
    }

    fn deal_one(&mut self) {
        self.dealer_hand.push(self.shoe.borrow_mut().one().unwrap());
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
    } else {
        Score::Hard(score)
    }
}

fn dealer_must_hit(score: Score) -> bool {
    match score {
        Score::Hard(score) => score < 17,
        Score::Soft(score) => score < 18,
    }
}

impl Score {
    pub fn value(&self) -> u8 {
        match self {
            Score::Hard(score) => *score,
            Score::Soft(score) => *score,
        }
    }

    pub fn blackjack(&self) -> bool {
        self.value() == 21
    }

    pub fn bust(&self) -> bool {
        match self {
            Score::Hard(score) => *score > 21,
            _ => false,
        }
    }
}
