use blackjack::{
    game::{
        card::{Card, Rank, Suit},
        game::{Game, Outcome, Score},
    },
    model::{
        Model,
        dqn::{Dqn, State},
    },
    training::{
        data::GameDataset,
        simulation::{Action, Simulation},
    },
};
use burn::{
    backend::{Candle, candle::CandleDevice},
    data::dataset::DatasetIterator,
    tensor::{Int, Tensor, cast::ToElement},
};
use strum::IntoEnumIterator;

use crate::basic_strategy::{HARD_TABLE, Rule, SOFT_TABLE, basic_strategy, should_split};

use super::basic_strategy::SPLIT_TABLE;

type B = Candle;
type D = CandleDevice;
type M = Dqn<B>;

pub struct Accuracy<'a> {
    model: &'a M,
    device: &'a D,
}

pub struct AccuracyReport {
    pub player: Score,
    pub dealer: Rank,
    pub action: Action,
    pub correct: Action,
    pub q: Tensor<B, 2>,
}

impl<'a> Accuracy<'a> {
    pub fn new(model: &'a M, device: &'a D) -> Self {
        Self { model, device }
    }

    pub fn eval_split(&self) -> Vec<AccuracyReport> {
        let mut reports = Vec::new();

        for item in &SPLIT_TABLE {
            for dealer in Rank::iter() {
                if !item.dealer.contains(&dealer) {
                    continue;
                }

                let player = if item.player != Rank::Ace {
                    Score::Hard(item.player.value() * 2)
                } else {
                    Score::Soft(12)
                };

                let (q, _, action) = forward(player, true, dealer, None, self.model, self.device);

                reports.push(AccuracyReport {
                    player,
                    dealer,
                    action,
                    correct: Action::Split,
                    q,
                });
            }
        }

        reports
    }

    pub fn eval_soft(&self) -> Vec<AccuracyReport> {
        self.eval(&SOFT_TABLE)
    }

    pub fn eval_hard(&self) -> Vec<AccuracyReport> {
        self.eval(&HARD_TABLE)
    }

    fn eval(&self, table: &[Rule]) -> Vec<AccuracyReport> {
        let mut reports = Vec::new();

        for item in table {
            for dealer in Rank::iter() {
                if !item.dealer.contains(&dealer) {
                    continue;
                }

                let (q, _, action) =
                    forward(item.player, false, dealer, None, self.model, self.device);

                reports.push(AccuracyReport {
                    player: item.player,
                    dealer,
                    action,
                    correct: item.action,
                    q,
                });
            }
        }

        reports
    }
}

pub struct ExpectedValue<'a> {
    model: &'a M,
    device: &'a D,
}

impl<'a> ExpectedValue<'a> {
    pub fn new(model: &'a M, device: &'a D) -> Self {
        Self { model, device }
    }

    pub fn eval_gameplay(&self, rounds: usize) -> (f32, f32) {
        let dataset = GameDataset::new(rounds, 6, false, None);
        let mut iter = DatasetIterator::new(&dataset);

        let mut model_win_loss = 0.0;
        let mut strat_win_loss = 0.0;
        for _ in 0..rounds {
            let mut game = iter.next().unwrap();

            let player = game.add_player(1.0);
            game.start();

            let mut model_game = self.play_as_model(game.clone(), player, None);
            let mut strat_game = self.play_as_strat(game.clone(), player, true);

            model_win_loss += Self::end_game(&mut model_game);
            strat_win_loss += Self::end_game(&mut strat_game)
        }

        (model_win_loss, strat_win_loss)
    }

    fn play_as_model(&self, game: Game, player: usize, prev_action: Option<Action>) -> Game {
        if !game.player_active(player) {
            return game;
        }

        let filter = if let Some(prev_action) = prev_action {
            match prev_action {
                Action::Split => Some(vec![
                    Action::Hit,
                    Action::Stand,
                    Action::Double,
                    Action::Split,
                ]),
                Action::Hit => Some(vec![Action::Hit, Action::Stand]),
                Action::Double | Action::Stand | Action::Surrender => panic!("Player not active."),
            }
        } else {
            None
        };

        let score = game.player_score(player);
        let can_split = game.player_can_split(player);
        let dealer = game.dealer_upcard().rank;
        let (_, _, action) = forward(score, can_split, dealer, filter, self.model, self.device);

        let mut game = game;
        if action == Action::Split {
            let split_player = game.player_split(player);

            game = self.play_as_model(game, player, Some(Action::Split));
            game = self.play_as_model(game, split_player, Some(Action::Split));
        } else {
            game = Simulation::new(game).forward(player, action);
        }

        self.play_as_model(game, player, Some(action))
    }

    fn play_as_strat(&self, game: Game, player: usize, first_action: bool) -> Game {
        if !game.player_active(player) {
            return game;
        }

        let mut game = game;

        if first_action && should_split(&game, player).is_some() {
            let split_player = game.player_split(player);

            game = self.play_as_strat(game, player, true);
            game = self.play_as_strat(game, split_player, true);

            return game;
        }
        let filter = if first_action {
            None
        } else {
            Some([Action::Hit, Action::Stand].as_ref())
        };

        let action = basic_strategy(&game, player, filter).unwrap();
        game = Simulation::new(game).forward(player, action);

        self.play_as_strat(game, player, false)
    }

    fn end_game(game: &mut Game) -> f32 {
        game.end();

        (0..game.player_count())
            .map(|player| match game.player_outcome(player).unwrap() {
                Outcome::PlayerWin(amount) => amount,
                Outcome::DealerWin(amount) => -amount,
                Outcome::Push => 0.0,
            })
            .sum()
    }
}

fn forward(
    player: Score,
    can_split: bool,
    dealer: Rank,
    filter: Option<Vec<Action>>,
    model: &M,
    device: &D,
) -> (Tensor<B, 2>, f32, Action) {
    let dealer = Card {
        rank: dealer,
        suit: Suit::Diamonds,
    };

    let state = State::raw_normalize(&player, can_split, &dealer);
    let state = Tensor::<B, 2>::from_data([state], device);

    let q = model.forward(state);
    let ev = q.clone().max_dim(1).into_scalar().to_f32();

    let filter = match filter {
        Some(filter) => filter,
        None => {
            let mut actions = vec![
                Action::Hit,
                Action::Stand,
                Action::Double,
                Action::Surrender,
            ];

            if can_split {
                actions.push(Action::Split);
            }

            actions
        }
    };
    let filter: Vec<_> = filter.iter().map(|&action| usize::from(action)).collect();
    let filter = Tensor::<B, 1, Int>::from_data(filter.as_slice(), device);

    let action = q
        .clone()
        .select(1, filter)
        .argmax(1)
        .into_scalar()
        .to_usize();

    (q, ev, Action::try_from(action).unwrap())
}
