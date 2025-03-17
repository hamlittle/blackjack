use std::{ops::RangeInclusive, path::Path};

use blackjack::{
    game::{
        card::{Card, Rank, Suit},
        game::{Game, Outcome, Score},
    },
    training::{
        data::{GameBatcher, GameDataset},
        model::{Action, Model},
        training::TrainingConfig,
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
use strum::IntoEnumIterator;

pub fn expected_value(
    player_score: Score,
    dealer_upcard: Card,
    model: &Model<Candle>,
    device: &CandleDevice,
) -> f32 {
    let state = Model::<Candle>::normalize(player_score, dealer_upcard, &device);
    let output = model.forward(state);
    output.max_dim(1).into_scalar().to_f32()
}

pub fn best_action(
    player_score: Score,
    dealer_upcard: Card,
    first_play: bool,
    model: &Model<Candle>,
    device: &CandleDevice,
) -> Action {
    let state = Model::<Candle>::normalize(player_score, dealer_upcard, &device);
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

pub fn evaluate(
    game: &mut Game,
    player: usize,
    terminal: bool,
    model: &Model<Candle>,
    device: &CandleDevice,
) -> (f32, Vec<Action>) {
    let ev = expected_value(
        game.player_score(player),
        game.dealer_upcard(),
        model,
        device,
    );

    // evaluate for hand split
    let player_hand = game.player_hand(0);
    if player_hand[0] == player_hand[1] {
        let split_ev = expected_value(
            Score::Hard(player_hand[0].rank.value()),
            game.dealer_upcard(),
            model,
            device,
        );

        // take the split
        if split_ev > ev {
            let split_player = game.player_split(player);

            let first_hand = evaluate(game, player, false, model, device);
            let second_hand = evaluate(game, split_player, true, model, device);

            let total_win_loss = first_hand.0 + second_hand.0;

            let mut actions = vec![Action::Split];
            actions.extend(first_hand.1);
            actions.extend(second_hand.1);

            return (total_win_loss, actions);
        }
    }

    // play the game normally
    let mut actions = Vec::new();
    loop {
        let action = best_action(
            game.player_score(0),
            game.dealer_upcard(),
            actions.len() == 0,
            model,
            device,
        );

        actions.push(action);

        match Action::try_from(action).unwrap() {
            Action::Hit => {
                game.player_hit(0);

                if game.player_outcome(0).is_some() {
                    break;
                }
            }
            Action::Stand => {
                game.player_stand(0);
                break;
            }
            Action::Double => {
                game.player_double(0);
                break;
            }
            Action::Surrender => {
                game.player_surrender(0);
                break;
            }
            Action::Split => panic!("ERR! Model does not evaluate splits."),
        }
    }

    if terminal {
        game.end();
    }

    let win_loss = match game.player_outcome(0).unwrap() {
        Outcome::PlayerWin(amount) => amount,
        Outcome::DealerWin(bet) => -bet,
        Outcome::Push => 0.0,
    };

    (win_loss, actions)
}

pub fn main() {
    let device = CandleDevice::default();

    let config = TrainingConfig::load(format!("/tmp/training/config.json"))
        .expect("Config should exist for the model.");

    let record = CompactRecorder::new()
        .load(format!("/tmp/training/model.mpk").into(), &device)
        // .load(format!("training/25_000h.10e/model.mpk").into(), &device)
        .expect("Trained model should exist.");

    let model = config.model.init::<Candle>(&device).load_record(record);

    let dataset = GameDataset::new(Path::new("out/shoes.1-25000.ndjson"), 10_000);
    let batcher = GameBatcher::<Candle>::new(device.clone());
    let loader = DataLoaderBuilder::new(batcher)
        .batch_size(1)
        .num_workers(1)
        .build(dataset);

    let mut total_win_loss = 0.0;
    let mut action_counts: Vec<usize> = vec![0; Action::iter().len()];
    for batch in loader.iter() {
        let mut game = batch.games[0].clone();
        let (win_loss, actions) = evaluate(&mut game, 0, true, &model, &device);

        total_win_loss += win_loss;

        for action in actions {
            action_counts[usize::from(action)] += 1;
        }
    }

    let hards = vec![
        // Hard hands
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

    println!("---");
    let hards = evaluate_accuracy("Hards", &hards, &model, &device);

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

            let state = Model::<Candle>::normalize(*player_score, dealer_upcard, &device);
            let q_values = model.forward(state);

            let chose_action = best_action(*player_score, dealer_upcard, true, &model, &device);

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
