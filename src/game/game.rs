use super::{
    card::{Card, Rank},
    shoe::Shoe,
};

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum GameStatus {
    PlaceBets,
    PlayerTurn,
    GameOver,
}

#[derive(Clone, Debug)]
pub struct Game {
    shoe: Shoe,
    dealer_hand: Vec<Card>,
    players: Vec<Player>,
    status: GameStatus,
}

#[derive(Clone, Debug)]
struct Player {
    hand: Vec<Card>,
    bet: f32,
    stand: bool,
    outcome: Option<Outcome>,
}

#[derive(Copy, Clone, Debug)]
pub enum Outcome {
    PlayerWin(f32),
    DealerWin(f32),
    Push,
}

#[derive(Copy, Clone, Debug)]
pub enum Score {
    Hard(u8),
    Soft(u8),
}

impl Game {
    pub fn new(shoe: Shoe) -> Self {
        Self {
            shoe,
            dealer_hand: Vec::new(),
            players: Vec::new(),
            status: GameStatus::PlaceBets,
        }
    }

    pub fn new_exact(shoe: Shoe, dealer_hand: Vec<Card>, players: Vec<(f32, Vec<Card>)>) -> Self {
        let players: Vec<_> = players
            .into_iter()
            .map(|(bet, cards)| {
                let mut player = Player::new(bet);
                player.hand = cards;

                player
            })
            .collect();

        Self {
            shoe,
            dealer_hand,
            players,
            status: GameStatus::PlaceBets,
        }
    }

    pub fn add_player(&mut self, bet: f32) -> usize {
        if self.status != GameStatus::PlaceBets {
            panic!("Cannot add player after game has started.");
        }

        self.players.push(Player::new(bet));

        self.players.len() - 1
    }

    pub fn player_count(&self) -> usize {
        self.players.len()
    }

    pub fn start(&mut self) -> GameStatus {
        if self.status != GameStatus::PlaceBets {
            panic!("Game has already started.");
        }

        while self.dealer_hand.len() < 2 {
            self.dealer_hand.push(self.shoe.one().unwrap());
        }

        for player in &mut self.players {
            while player.hand.len() < 2 {
                player.hand.push(self.shoe.one().unwrap());
            }
        }

        if score(&self.dealer_hand).blackjack() {
            self.players.iter_mut().for_each(|player| {
                player.action_dealer_blackjack();
            });

            self.status = GameStatus::GameOver;
        } else {
            self.players.iter_mut().for_each(|player| {
                if score(&player.hand).blackjack() {
                    player.action_player_blackjack();
                }
            });

            if self.players.iter().all(|player| player.stand) {
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

        if !self.players.iter().all(|player| player.stand) {
            panic!("Cannot end game until all players stand.");
        }

        while dealer_must_hit(score(&self.dealer_hand)) {
            self.dealer_hand.push(self.shoe.one().unwrap());
        }

        let dealer_score = score(&self.dealer_hand);

        self.players.iter_mut().for_each(|player| {
            player.action_end(&dealer_score);
        });

        self.status = GameStatus::GameOver;
    }

    pub fn player_hit(&mut self, player: usize) -> Option<Outcome> {
        self.players[player].action_hit(self.shoe.one().unwrap())
    }

    pub fn player_stand(&mut self, player: usize) {
        self.players[player].action_stand();
    }

    pub fn player_double(&mut self, player: usize) -> Option<Outcome> {
        self.players[player].action_double(self.shoe.one().unwrap())
    }

    pub fn player_split(&mut self, player: usize) -> usize {
        if self.status != GameStatus::PlayerTurn {
            panic!("Cannot split unless player turn.");
        }

        let player = &mut self.players[player];
        let card = player.action_split();
        player.hand.push(self.shoe.one().unwrap());

        let mut split_play = Player::new(player.bet);
        split_play.hand.push(card);
        split_play.hand.push(self.shoe.one().unwrap());

        self.players.push(split_play);

        self.players.len() - 1
    }

    pub fn player_surrender(&mut self, player: usize) -> Outcome {
        self.players[player].action_surrender()
    }

    pub fn player_hand(&self, player: usize) -> Vec<Card> {
        self.players[player].hand.clone()
    }

    pub fn dealer_hand(&self) -> Vec<Card> {
        self.dealer_hand.clone()
    }

    pub fn player_bet(&self, player: usize) -> f32 {
        self.players[player].bet
    }

    pub fn player_score(&self, player: usize) -> Score {
        score(&self.players[player].hand)
    }

    pub fn player_outcome(&self, player: usize) -> Option<Outcome> {
        self.players[player].outcome
    }

    pub fn dealer_upcard(&self) -> Card {
        self.dealer_hand[0]
    }

    pub fn shoe(&self) -> Shoe {
        self.shoe.clone()
    }
}

impl Player {
    fn new(bet: f32) -> Self {
        Self {
            hand: Vec::new(),
            bet,
            stand: false,
            outcome: None,
        }
    }

    fn action_hit(&mut self, card: Card) -> Option<Outcome> {
        if self.stand {
            panic!("Cannot hit after standing.");
        }

        self.hand.push(card);

        if score(&self.hand).bust() {
            self.outcome = Some(Outcome::DealerWin(self.bet));
            self.stand = true;
        }

        self.outcome
    }

    fn action_stand(&mut self) {
        self.stand = true;
    }

    fn action_double(&mut self, card: Card) -> Option<Outcome> {
        if self.stand {
            panic!("Cannot double after standing.");
        }

        if self.hand.len() != 2 {
            panic!("Cannot double after taking an action.");
        }

        self.bet *= 2.0;
        self.action_hit(card);
        self.action_stand();

        self.outcome
    }

    fn action_split(&mut self) -> Card {
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

    pub fn action_surrender(&mut self) -> Outcome {
        if self.stand {
            panic!("Cannot surrender after standing.");
        }

        if self.hand.len() != 2 {
            panic!("Cannot surrender after taking an action.");
        }

        self.outcome = Some(Outcome::DealerWin(self.bet / 2.0));
        self.stand = true;

        self.outcome.unwrap()
    }

    fn action_dealer_blackjack(&mut self) {
        if score(&self.hand).blackjack() {
            self.outcome = Some(Outcome::Push);
        } else {
            self.outcome = Some(Outcome::DealerWin(self.bet));
        }

        self.stand = true;
    }

    fn action_player_blackjack(&mut self) {
        self.outcome = Some(Outcome::PlayerWin(self.bet * 1.5));
        self.stand = true;
    }

    fn action_end(&mut self, dealer_score: &Score) {
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

        match score(&self.hand).value().cmp(&dealer_score.value()) {
            std::cmp::Ordering::Less => self.outcome = Some(Outcome::DealerWin(self.bet)),
            std::cmp::Ordering::Equal => self.outcome = Some(Outcome::Push),
            std::cmp::Ordering::Greater => self.outcome = Some(Outcome::PlayerWin(self.bet)),
        }
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
