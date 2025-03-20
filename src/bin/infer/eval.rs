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

use crate::basic_strategy::{
    HARD_TABLE, Rule, SOFT_TABLE, basic_strategy, should_hit, should_split,
};

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

                let (q, ev, action) = forward(player, dealer, None, self.model, self.device);

                let split_player = Score::Hard(item.player.value());
                let filter = [Action::Hit];
                let (_, split_ev, _) =
                    forward(split_player, dealer, Some(&filter), self.model, self.device);

                let action = if split_ev > ev { Action::Split } else { action };

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

                let (q, _, action) = forward(item.player, dealer, None, self.model, self.device);

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
        let dataset = GameDataset::new(rounds, 52);
        let mut iter = DatasetIterator::new(&dataset);

        let mut model_win_loss = 0.0;
        let mut strat_win_loss = 0.0;
        for _ in 0..rounds {
            let game = iter.next().unwrap();

            let mut model_game = self.play_as_model(game.clone(), 0, true);
            let mut strat_game = self.play_as_strat(game.clone(), 0, true);

            model_win_loss += Self::end_game(&mut model_game);
            strat_win_loss += Self::end_game(&mut strat_game)
        }

        (model_win_loss, strat_win_loss)
    }

    fn play_as_model(&self, game: Game, player: usize, first_action: bool) -> Game {
        let mut game = game;

        if first_action && self.model_should_split(&game, player) {
            let split_player = game.player_split(player);

            game = self.play_as_model(game, player, true);
            game = self.play_as_model(game, split_player, true);

            return game;
        }

        let mut first_action = first_action;
        while game.player_active(player) {
            let score = game.player_score(player);
            let dealer = game.dealer_upcard().rank;

            let filter = if first_action {
                None
            } else {
                Some([Action::Hit, Action::Stand].as_ref())
            };

            let (_, _, action) = forward(score, dealer, filter, self.model, self.device);
            game = Simulation::new(game).forward(player, action);

            first_action = false;
        }

        game
    }

    fn play_as_strat(&self, game: Game, player: usize, first_action: bool) -> Game {
        let mut game = game;

        if first_action && should_split(&game, player).is_some() {
            let split_player = game.player_split(player);

            game = self.play_as_strat(game, player, true);
            game = self.play_as_strat(game, split_player, true);

            return game;
        }

        let mut first_action = first_action;
        while game.player_active(player) {
            let filter = if first_action {
                None
            } else {
                Some([Action::Hit, Action::Stand].as_ref())
            };

            let action = basic_strategy(&game, player, filter).unwrap();
            game = Simulation::new(game).forward(player, action);

            first_action = false;
        }

        game
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

    fn model_should_split(&self, game: &Game, player: usize) -> bool {
        let player_hand = game.player_hand(player);
        let dealer = game.dealer_upcard();

        if player_hand[0] != player_hand[1] {
            return false;
        }

        let player = game.player_score(player);
        let dealer = dealer.rank;
        let (_, ev, _) = forward(player, dealer, None, self.model, self.device);

        let split_player = Score::Hard(player.value() / 2);
        let (_, split_ev, _) = forward(
            split_player,
            dealer,
            Some(&[Action::Hit]),
            self.model,
            self.device,
        );

        return split_ev > ev;
    }
}

fn forward(
    player: Score,
    dealer: Rank,
    filter: Option<&[Action]>,
    model: &M,
    device: &D,
) -> (Tensor<B, 2>, f32, Action) {
    let dealer = Card {
        rank: dealer,
        suit: Suit::Diamonds,
    };

    let state = State::raw_normalize(&player, &dealer);
    let state = Tensor::<B, 2>::from_data([state], device);

    let q = model.forward(state);
    let ev = q.clone().max_dim(1).into_scalar().to_f32();

    let filter = match filter {
        Some(filter) => filter,
        None => &[
            Action::Hit,
            Action::Stand,
            Action::Double,
            Action::Surrender,
        ],
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

// pub fn expected_value(
//     player_score: Score,
//     dealer_upcard: Card,
//     filter: Option<&[Action]>,
//     model: &M,
//     device: &D,
// ) -> f32 {
//     let state = State::raw_normalize(&player_score, &dealer_upcard);
//     let state = Tensor::<B, 2>::from_data([state], device);
//     let q = model.forward(state);

//     let filter: Vec<_> = match filter {
//         Some(filter) => filter.iter().map(|action| *action as u32).collect(),
//         None => (0..M::output_size() as u32).collect(),
//     };

//     let filter = Tensor::<Candle, 1, Int>::from_data(filter.as_slice(), &device).reshape([1, -1]);
//     q.gather(1, filter).max_dim(1).into_scalar().to_f32()
// }

// pub fn best_action(
//     player_score: Score,
//     dealer_upcard: Card,
//     first_play: bool,
//     can_split: bool,
//     model: &M,
//     device: &CandleDevice,
// ) -> Action {
//     if first_play && can_split {
//         let ev = expected_value(player_score, dealer_upcard, None, model, device);

//         let split_ev = expected_value(
//             Score::Hard(player_score.value() / 2),
//             dealer_upcard,
//             Some(&[Action::Hit]),
//             model,
//             device,
//         ) * 2.0;

//         if split_ev > ev {
//             return Action::Split;
//         }
//     }

//     let state = State::raw_normalize(&player_score, &dealer_upcard);
//     let state = Tensor::<B, 2>::from_data([state], device);
//     let q = model.forward(state);

//     let q = if !first_play {
//         let filter = [Action::Hit as u32, Action::Stand as u32];
//         let filter = Tensor::<Candle, 2, Int>::from_data([filter], &device);

//         q.gather(1, filter)
//     } else {
//         q
//     };

//     let best_action = q.argmax(1).into_scalar().to_usize();

//     Action::try_from(best_action).unwrap()
// }

// pub fn evaluate_gameplay(
//     game: &mut Game,
//     player: usize,
//     terminal: bool,
//     is_split: bool,
//     model: &M,
//     device: &CandleDevice,
// ) -> (f32, Action) {
//     let player_hand = game.player_hand(0);
//     let first_action = best_action(
//         game.player_score(0),
//         game.dealer_upcard(),
//         true,
//         player_hand[0].rank == player_hand[1].rank,
//         model,
//         device,
//     );

//     if first_action == Action::Split {
//         let split_player = game.player_split(player);
//         let _ = evaluate_gameplay(game, player, false, true, model, device);
//         let _ = evaluate_gameplay(game, split_player, false, true, model, device);

//         let mut win_loss = 0.0;
//         if !is_split {
//             game.end();

//             win_loss = (0..game.player_count())
//                 .map(|player| match game.player_outcome(player).unwrap() {
//                     Outcome::PlayerWin(amount) => amount,
//                     Outcome::DealerWin(bet) => -bet,
//                     Outcome::Push => 0.0,
//                 })
//                 .sum();
//         }

//         return (win_loss, Action::Split);
//     } else {
//         let mut action = first_action;

//         while game.player_outcome(player).is_none() {
//             match action {
//                 Action::Hit => {
//                     game.player_hit(player);
//                 }
//                 Action::Stand => {
//                     game.player_stand(player);
//                     break;
//                 }
//                 Action::Double => {
//                     game.player_double(player);
//                     break;
//                 }
//                 Action::Surrender => {
//                     game.player_surrender(player);
//                     break;
//                 }
//                 Action::Split => panic!("ERR! Model does not evaluate splits."),
//             }

//             action = best_action(
//                 game.player_score(player),
//                 game.dealer_upcard(),
//                 false,
//                 false,
//                 model,
//                 device,
//             );
//         }
//     }

//     let mut win_loss = 0.0;
//     if terminal {
//         game.end();

//         win_loss = match game.player_outcome(player).unwrap() {
//             Outcome::PlayerWin(amount) => amount,
//             Outcome::DealerWin(bet) => -bet,
//             Outcome::Push => 0.0,
//         };
//     }

//     (win_loss, first_action)
// }

// fn evaluate_accuracy(
//     test_type: &str,
//     test_data: &[(Score, RangeInclusive<Rank>, Action)],
//     eval_split: bool,
//     model: &M,
//     device: &CandleDevice,
// ) -> (u32, u32, f32) {
//     let mut correct_plays = 0;
//     let mut incorrect_plays = 0;

//     for (player_score, dealer_upcard, correct_action) in test_data {
//         for dealer_upcard in Rank::iter().filter(|r| dealer_upcard.contains(r)) {
//             let dealer_upcard = Card {
//                 rank: dealer_upcard,
//                 suit: Suit::Diamonds,
//             };

//             let action = best_action(
//                 *player_score,
//                 dealer_upcard,
//                 true,
//                 eval_split,
//                 model,
//                 device,
//             );

//             if action == *correct_action {
//                 correct_plays += 1;
//             } else {
//                 incorrect_plays += 1;
//             }

//             let state = State::raw_normalize(player_score, &dealer_upcard);
//             let state = Tensor::<B, 2>::from_data([state], device);
//             let q = model.forward(state);

//             println!(
//                 "{} {} | {:?} vs {}: Should {:?}, Chose {:?} -> Q-values: {}",
//                 if action == *correct_action {
//                     "\u{2705}"
//                 } else {
//                     "\u{274C}"
//                 },
//                 test_type,
//                 player_score,
//                 dealer_upcard.rank.value(),
//                 correct_action,
//                 action,
//                 q.into_data()
//             )
//         }
//     }

//     (
//         correct_plays,
//         incorrect_plays,
//         correct_plays as f32 / (correct_plays + incorrect_plays) as f32,
//     )
// }
