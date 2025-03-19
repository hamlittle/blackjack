use std::{ops::RangeInclusive, path::Path};

use blackjack::{
    game::{
        card::{Card, Rank, Suit},
        game::{Game, Outcome, Score},
    },
    training::{
        data::GameDataset,
        ppo::{Model, State},
        simulation::Action,
    },
};
use burn::{
    backend::{Candle, candle::CandleDevice},
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{Int, Tensor, cast::ToElement},
};
use data::ModelBatcher;
use strum::IntoEnumIterator;
use train::TrainingConfig;

mod data;
mod metrics;
mod render;
mod train;

pub fn expected_value(
    player_score: Score,
    dealer_upcard: Card,
    filter: Option<&[Action]>,
    model: &Model<Candle>,
    device: &CandleDevice,
) -> f32 {
    let state = Model::<Candle>::normalize(
        &[State {
            player_score,
            dealer_upcard,
        }],
        &device,
    );
    let output = model.forward(state);

    let filter: Vec<usize> = match filter {
        Some(filter) => filter.iter().map(|action| usize::from(*action)).collect(),
        None => (0..Model::<Candle>::output_size()).collect(),
    };
    let filter = Tensor::<Candle, 1, Int>::from_data(filter.as_slice(), &device);

    output.select(1, filter).max_dim(1).into_scalar().to_f32()
}

pub fn best_action(
    player_score: Score,
    dealer_upcard: Card,
    first_play: bool,
    can_split: bool,
    model: &Model<Candle>,
    device: &CandleDevice,
) -> Action {
    // if first_play && can_split {
    //     let ev = expected_value(player_score, dealer_upcard, None, model, device);

    //     let split_ev = expected_value(
    //         Score::Hard(player_score.value() / 2),
    //         dealer_upcard,
    //         Some(&[Action::Hit, Action::Stand]),
    //         model,
    //         device,
    //     ) * 2.0;

    //     if split_ev > ev {
    //         return Action::Split;
    //     }
    // }

    let state = Model::<Candle>::normalize(
        &[State {
            player_score,
            dealer_upcard,
        }],
        &device,
    );
    let output = model.forward(state);

    let output = if !first_play {
        let filter = [Action::Hit, Action::Stand].map(usize::from);
        let filter = Tensor::<Candle, 1, Int>::from_data(filter, &device);

        output.select(1, filter)
    } else {
        output
    };

    let best_action = output.argmax(1).into_scalar().to_usize();

    Action::try_from(best_action).unwrap()
}

pub fn evaluate_gameplay(
    game: &mut Game,
    player: usize,
    terminal: bool,
    is_split: bool,
    model: &Model<Candle>,
    device: &CandleDevice,
) -> (f32, Action) {
    let player_hand = game.player_hand(0);
    let first_action = best_action(
        game.player_score(0),
        game.dealer_upcard(),
        true,
        player_hand[0].rank == player_hand[1].rank,
        model,
        device,
    );

    if first_action == Action::Split {
        let split_player = game.player_split(player);
        let _ = evaluate_gameplay(game, player, false, true, model, device);
        let _ = evaluate_gameplay(game, split_player, false, true, model, device);

        let mut win_loss = 0.0;
        if !is_split {
            game.end();

            win_loss = (0..game.player_count())
                .map(|player| match game.player_outcome(player).unwrap() {
                    Outcome::PlayerWin(amount) => amount,
                    Outcome::DealerWin(bet) => -bet,
                    Outcome::Push => 0.0,
                })
                .sum();
        }

        return (win_loss, Action::Split);
    } else {
        let mut action = first_action;

        while game.player_outcome(player).is_none() {
            match action {
                Action::Hit => {
                    game.player_hit(player);
                }
                Action::Stand => {
                    game.player_stand(player);
                    break;
                }
                Action::Double => {
                    game.player_double(player);
                    break;
                }
                Action::Surrender => {
                    game.player_surrender(player);
                    break;
                }
                Action::Split => panic!("ERR! Model does not evaluate splits."),
            }

            action = best_action(
                game.player_score(player),
                game.dealer_upcard(),
                false,
                false,
                model,
                device,
            );
        }
    }

    let mut win_loss = 0.0;
    if terminal {
        game.end();

        win_loss = match game.player_outcome(player).unwrap() {
            Outcome::PlayerWin(amount) => amount,
            Outcome::DealerWin(bet) => -bet,
            Outcome::Push => 0.0,
        };
    }

    (win_loss, first_action)
}

pub fn main() {
    let device = CandleDevice::default();

    // let dir = "training/best";
    let dir = "/tmp/training";
    let model = "model";
    let rounds = 10_000;

    let config = TrainingConfig::load(format!("{dir}/config.json"))
        .expect("Config should exist for the model.");

    let record = CompactRecorder::new()
        .load(format!("{dir}/{model}.mpk").into(), &device)
        .expect("Trained model should exist.");

    let model = config.model.init::<Candle>(&device).load_record(record);

    let dataset = GameDataset::new(rounds);
    let batcher = ModelBatcher::<Candle>::new(device.clone());
    let loader = DataLoaderBuilder::new(batcher)
        .batch_size(1)
        .num_workers(1)
        .build(dataset);

    let mut total_win_loss = 0.0;
    let mut action_counts: Vec<usize> = vec![0; Action::iter().len()];
    for batch in loader.iter() {
        let mut game = batch.games[0].clone();
        let (win_loss, first_action) =
            evaluate_gameplay(&mut game, 0, true, false, &model, &device);

        total_win_loss += win_loss;
        action_counts[usize::from(first_action)] += 1;
    }

    // let splits = vec![
    //     // (Score::Hard(4), Rank::Two..=Rank::Seven, Action::Split),
    //     // (Score::Hard(4), Rank::Eight..=Rank::Ace, Action::Hit),
    //     // (Score::Hard(6), Rank::Two..=Rank::Seven, Action::Split),
    //     // (Score::Hard(6), Rank::Eight..=Rank::Ace, Action::Hit),
    //     (Score::Hard(8), Rank::Two..=Rank::Four, Action::Split),
    //     (Score::Hard(8), Rank::Five..=Rank::Six, Action::Split),
    //     (Score::Hard(8), Rank::Six..=Rank::Ace, Action::Hit),
    //     (Score::Hard(10), Rank::Two..=Rank::Nine, Action::Double),
    //     (Score::Hard(10), Rank::Ten..=Rank::Ace, Action::Hit),
    //     (Score::Hard(12), Rank::Two..=Rank::Six, Action::Split),
    //     (Score::Hard(12), Rank::Seven..=Rank::Ace, Action::Hit),
    //     (Score::Hard(14), Rank::Two..=Rank::Seven, Action::Split),
    //     (Score::Hard(14), Rank::Eight..=Rank::Ace, Action::Hit),
    //     (Score::Hard(16), Rank::Two..=Rank::Ten, Action::Split),
    //     (Score::Hard(16), Rank::Ace..=Rank::Ace, Action::Surrender),
    //     (Score::Hard(18), Rank::Two..=Rank::Six, Action::Split),
    //     (Score::Hard(18), Rank::Seven..=Rank::Seven, Action::Stand),
    //     (Score::Hard(18), Rank::Eight..=Rank::Nine, Action::Split),
    //     (Score::Hard(18), Rank::Ten..=Rank::Ace, Action::Stand),
    //     (Score::Hard(20), Rank::Two..=Rank::Ace, Action::Stand),
    //     // (Score::Hard(22), Rank::Two..=Rank::Ace, Action::Split),
    // ];

    let softs = vec![
        (Score::Soft(13), Rank::Two..=Rank::Four, Action::Hit),
        (Score::Soft(13), Rank::Five..=Rank::Six, Action::Double),
        (Score::Soft(13), Rank::Seven..=Rank::Ace, Action::Hit),
        (Score::Soft(14), Rank::Two..=Rank::Four, Action::Hit),
        (Score::Soft(14), Rank::Five..=Rank::Six, Action::Double),
        (Score::Soft(14), Rank::Seven..=Rank::Ace, Action::Hit),
        (Score::Soft(15), Rank::Two..=Rank::Three, Action::Hit),
        (Score::Soft(15), Rank::Four..=Rank::Six, Action::Double),
        (Score::Soft(15), Rank::Seven..=Rank::Ace, Action::Hit),
        (Score::Soft(16), Rank::Two..=Rank::Three, Action::Hit),
        (Score::Soft(16), Rank::Four..=Rank::Six, Action::Double),
        (Score::Soft(16), Rank::Seven..=Rank::Ace, Action::Hit),
        (Score::Soft(17), Rank::Two..=Rank::Two, Action::Hit),
        (Score::Soft(17), Rank::Three..=Rank::Six, Action::Double),
        (Score::Soft(17), Rank::Seven..=Rank::Ace, Action::Hit),
        (Score::Soft(18), Rank::Two..=Rank::Six, Action::Double),
        (Score::Soft(18), Rank::Seven..=Rank::Eight, Action::Stand),
        (Score::Soft(18), Rank::Nine..=Rank::Ace, Action::Hit),
        (Score::Soft(19), Rank::Two..=Rank::Five, Action::Stand),
        (Score::Soft(19), Rank::Six..=Rank::Six, Action::Double),
        (Score::Soft(19), Rank::Seven..=Rank::Ace, Action::Stand),
        (Score::Soft(20), Rank::Two..=Rank::Ace, Action::Stand),
    ];

    let hards = vec![
        (Score::Hard(4), Rank::Two..=Rank::Ace, Action::Hit),
        (Score::Hard(5), Rank::Two..=Rank::Ace, Action::Hit),
        (Score::Hard(6), Rank::Two..=Rank::Ace, Action::Hit),
        (Score::Hard(7), Rank::Two..=Rank::Ace, Action::Hit),
        (Score::Hard(8), Rank::Two..=Rank::Ace, Action::Hit),
        (Score::Hard(9), Rank::Two..=Rank::Three, Action::Hit),
        (Score::Hard(9), Rank::Four..=Rank::Six, Action::Double),
        (Score::Hard(9), Rank::Seven..=Rank::Ace, Action::Hit),
        (Score::Hard(10), Rank::Two..=Rank::Nine, Action::Double),
        (Score::Hard(10), Rank::Ten..=Rank::Ace, Action::Hit),
        (Score::Hard(11), Rank::Two..=Rank::Ace, Action::Double),
        (Score::Hard(12), Rank::Two..=Rank::Three, Action::Hit),
        (Score::Hard(12), Rank::Four..=Rank::Six, Action::Stand),
        (Score::Hard(12), Rank::Seven..=Rank::Ace, Action::Hit),
        (Score::Hard(13), Rank::Two..=Rank::Six, Action::Stand),
        (Score::Hard(13), Rank::Seven..=Rank::Ace, Action::Hit),
        (Score::Hard(14), Rank::Two..=Rank::Six, Action::Stand),
        (Score::Hard(14), Rank::Seven..=Rank::Ace, Action::Hit),
        (Score::Hard(15), Rank::Two..=Rank::Six, Action::Stand),
        (Score::Hard(15), Rank::Seven..=Rank::Nine, Action::Hit),
        (Score::Hard(15), Rank::Ten..=Rank::Ace, Action::Surrender),
        (Score::Hard(16), Rank::Two..=Rank::Six, Action::Stand),
        (Score::Hard(16), Rank::Seven..=Rank::Eight, Action::Hit),
        (Score::Hard(16), Rank::Nine..=Rank::Ace, Action::Surrender),
        (Score::Hard(17), Rank::Two..=Rank::Ten, Action::Stand),
        (Score::Hard(17), Rank::Ace..=Rank::Ace, Action::Surrender),
        (Score::Hard(18), Rank::Two..=Rank::Ace, Action::Stand),
        (Score::Hard(19), Rank::Two..=Rank::Ace, Action::Stand),
        (Score::Hard(20), Rank::Two..=Rank::Ace, Action::Stand),
    ];

    // println!("---");
    // let splits = evaluate_accuracy("Split", &splits, true, &model, &device);
    println!("---");
    let softs = evaluate_accuracy("Soft", &softs, false, &model, &device);
    println!("---");
    let hards = evaluate_accuracy("Hard", &hards, false, &model, &device);

    println!("---");
    println!(
        "Total games: {}, Total win/loss: {}, EV: {:.3} %",
        loader.num_items(),
        total_win_loss,
        total_win_loss as f32 * 100.0 / loader.num_items() as f32
    );
    println!(
        "Hit {}, Stand {}, Double {}, Surrender {}, Split {}",
        action_counts[usize::from(Action::Hit)],
        action_counts[usize::from(Action::Stand)],
        action_counts[usize::from(Action::Double)],
        action_counts[usize::from(Action::Surrender)],
        action_counts[usize::from(Action::Split)]
    );
    // println!(
    //     "Splits: {} / {} -> {:.1} %",
    //     splits.0,
    //     splits.0 + splits.1,
    //     splits.2 * 100.0
    // );
    println!(
        "Softs: {} / {} -> {:.1} %",
        softs.0,
        softs.0 + softs.1,
        softs.2 * 100.0
    );
    println!(
        "Hards: {} / {} -> {:.1} %",
        hards.0,
        hards.0 + hards.1,
        hards.2 * 100.0
    );
}

fn evaluate_accuracy(
    test_type: &str,
    test_data: &[(Score, RangeInclusive<Rank>, Action)],
    eval_split: bool,
    model: &Model<Candle>,
    device: &CandleDevice,
) -> (u32, u32, f32) {
    let mut correct_plays = 0;
    let mut incorrect_plays = 0;

    for (player_score, dealer_upcard, correct_action) in test_data {
        for dealer_upcard in Rank::iter().filter(|r| dealer_upcard.contains(r)) {
            let dealer_upcard = Card {
                rank: Rank::try_from(dealer_upcard).unwrap(),
                suit: Suit::Diamonds,
            };

            let state = Model::<Candle>::normalize(
                &[State {
                    player_score: *player_score,
                    dealer_upcard,
                }],
                &device,
            );
            let q_values = model.forward(state);

            let chose_action = best_action(
                *player_score,
                dealer_upcard,
                true,
                eval_split,
                &model,
                &device,
            );

            if chose_action == *correct_action {
                correct_plays += 1;
            } else {
                incorrect_plays += 1;
            }

            println!(
                "{} {} | {:?} vs {}: Should {:?}, Chose {:?} -> Q-values: {}",
                if chose_action == *correct_action {
                    "\u{2705}"
                } else {
                    "\u{274C}"
                },
                test_type,
                player_score,
                dealer_upcard.rank.value(),
                correct_action,
                chose_action,
                q_values.into_data()
            )
        }
    }

    (
        correct_plays,
        incorrect_plays,
        correct_plays as f32 / (correct_plays + incorrect_plays) as f32,
    )
}
