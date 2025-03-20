use blackjack::model::dqn::{Dqn, ModelConfig};
use burn::{
    backend::{Candle, candle::CandleDevice},
    config::Config,
    module::Module,
    record::{CompactRecorder, Recorder},
};
use eval::{Accuracy, ExpectedValue};
use print::{print_accuracy, print_summary};

mod basic_strategy;
mod eval;
mod print;

type B = Candle;
type D = CandleDevice;
type M = Dqn<B>;

fn load(dir: &str, model: &str, device: &D) -> M {
    let config =
        ModelConfig::load(format!("{dir}/model.json")).expect("Config should exist for the model.");

    let params = CompactRecorder::new()
        .load(format!("{dir}/{model}.mpk").into(), device)
        .expect("Trained model should exist.");

    config.init::<B>(&device).load_record(params)
}

pub fn main() {
    let device = CandleDevice::default();

    let dir = "/tmp/training";
    let model = "model-7";
    let verbose = false;
    let rounds = 50_000;

    let model = load(dir, model, &device);

    let acc = Accuracy::new(&model, &device);
    let split_reports = acc.eval_split();
    let hard_reports = acc.eval_hard();
    let soft_reports = acc.eval_soft();

    let ev = ExpectedValue::new(&model, &device);
    let (model_ev, strat_ev) = ev.eval_gameplay(rounds);

    print_accuracy("Splits", &split_reports, verbose);
    print_accuracy("Soft", &soft_reports, verbose);
    print_accuracy("Hard", &hard_reports, verbose);

    println!("-- Summary");
    print_summary("Splits", &split_reports);
    print_summary("Soft", &soft_reports);
    print_summary("hard", &hard_reports);

    println!("-- % EV");
    println!("{} -> {:2} %", rounds, model_ev * 100.0 / rounds as f32);
    println!("{} -> {:2} %", rounds, strat_ev * 100.0 / rounds as f32);

    // let dataset = GameDataset::new(rounds, 52);
    // let mut iter = DatasetIterator::new(&dataset);

    // print!("---")
    // eval_splits()

    // let mut total_win_loss = 0.0;
    // let mut action_counts: Vec<usize> = vec![0; Action::iter().len()];
    // for _ in 0..rounds {
    //     let mut game = iter.next().unwrap();
    //     let (win_loss, first_action) =
    //         evaluate_gameplay(&mut game, 0, true, false, &model, &device);

    //     total_win_loss += win_loss;
    //     action_counts[usize::from(first_action)] += 1;
    // }

    // let splits = vec![
    //     // (Score::Hard(4), Rank::Two..=Rank::Seven, Action::Split),
    //     // (Score::Hard(4), Rank::Eight..=Rank::Ace, Action::Hit),
    //     // (Score::Hard(6), Rank::Two..=Rank::Seven, Action::Split),
    //     // (Score::Hard(6), Rank::Eight..=Rank::Ace, Action::Hit),
    //     (Score::Hard(8), Rank::Two..=Rank::Four, Action::Hit),
    //     (Score::Hard(8), Rank::Five..=Rank::Six, Action::Split),
    //     (Score::Hard(8), Rank::Seven..=Rank::Ace, Action::Hit),
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
    //     // (Score::Soft(12), Rank::Two..=Rank::Ace, Action::Split),
    // ];

    // let softs = vec![
    //     (Score::Soft(13), Rank::Two..=Rank::Four, Action::Hit),
    //     (Score::Soft(13), Rank::Five..=Rank::Six, Action::Double),
    //     (Score::Soft(13), Rank::Seven..=Rank::Ace, Action::Hit),
    //     (Score::Soft(14), Rank::Two..=Rank::Four, Action::Hit),
    //     (Score::Soft(14), Rank::Five..=Rank::Six, Action::Double),
    //     (Score::Soft(14), Rank::Seven..=Rank::Ace, Action::Hit),
    //     (Score::Soft(15), Rank::Two..=Rank::Three, Action::Hit),
    //     (Score::Soft(15), Rank::Four..=Rank::Six, Action::Double),
    //     (Score::Soft(15), Rank::Seven..=Rank::Ace, Action::Hit),
    //     (Score::Soft(16), Rank::Two..=Rank::Three, Action::Hit),
    //     (Score::Soft(16), Rank::Four..=Rank::Six, Action::Double),
    //     (Score::Soft(16), Rank::Seven..=Rank::Ace, Action::Hit),
    //     (Score::Soft(17), Rank::Two..=Rank::Two, Action::Hit),
    //     (Score::Soft(17), Rank::Three..=Rank::Six, Action::Double),
    //     (Score::Soft(17), Rank::Seven..=Rank::Ace, Action::Hit),
    //     (Score::Soft(18), Rank::Two..=Rank::Six, Action::Double),
    //     (Score::Soft(18), Rank::Seven..=Rank::Eight, Action::Stand),
    //     (Score::Soft(18), Rank::Nine..=Rank::Ace, Action::Hit),
    //     (Score::Soft(19), Rank::Two..=Rank::Five, Action::Stand),
    //     (Score::Soft(19), Rank::Six..=Rank::Six, Action::Double),
    //     (Score::Soft(19), Rank::Seven..=Rank::Ace, Action::Stand),
    //     (Score::Soft(20), Rank::Two..=Rank::Ace, Action::Stand),
    // ];

    // let hards = vec![
    //     (Score::Hard(4), Rank::Two..=Rank::Ace, Action::Hit),
    //     (Score::Hard(5), Rank::Two..=Rank::Ace, Action::Hit),
    //     (Score::Hard(6), Rank::Two..=Rank::Ace, Action::Hit),
    //     (Score::Hard(7), Rank::Two..=Rank::Ace, Action::Hit),
    //     (Score::Hard(8), Rank::Two..=Rank::Ace, Action::Hit),
    //     (Score::Hard(9), Rank::Two..=Rank::Two, Action::Hit),
    //     (Score::Hard(9), Rank::Three..=Rank::Six, Action::Double),
    //     (Score::Hard(9), Rank::Seven..=Rank::Ace, Action::Hit),
    //     (Score::Hard(10), Rank::Two..=Rank::Nine, Action::Double),
    //     (Score::Hard(10), Rank::Ten..=Rank::Ace, Action::Hit),
    //     (Score::Hard(11), Rank::Two..=Rank::Ace, Action::Double),
    //     (Score::Hard(12), Rank::Two..=Rank::Three, Action::Hit),
    //     (Score::Hard(12), Rank::Four..=Rank::Six, Action::Stand),
    //     (Score::Hard(12), Rank::Seven..=Rank::Ace, Action::Hit),
    //     (Score::Hard(13), Rank::Two..=Rank::Six, Action::Stand),
    //     (Score::Hard(13), Rank::Seven..=Rank::Ace, Action::Hit),
    //     (Score::Hard(14), Rank::Two..=Rank::Six, Action::Stand),
    //     (Score::Hard(14), Rank::Seven..=Rank::Ace, Action::Hit),
    //     (Score::Hard(15), Rank::Two..=Rank::Six, Action::Stand),
    //     (Score::Hard(15), Rank::Seven..=Rank::Nine, Action::Hit),
    //     (Score::Hard(15), Rank::Ten..=Rank::Ace, Action::Surrender),
    //     (Score::Hard(16), Rank::Two..=Rank::Six, Action::Stand),
    //     (Score::Hard(16), Rank::Seven..=Rank::Eight, Action::Hit),
    //     (Score::Hard(16), Rank::Nine..=Rank::Ace, Action::Surrender),
    //     (Score::Hard(17), Rank::Two..=Rank::Ten, Action::Stand),
    //     (Score::Hard(17), Rank::Ace..=Rank::Ace, Action::Surrender),
    //     (Score::Hard(18), Rank::Two..=Rank::Ace, Action::Stand),
    //     (Score::Hard(19), Rank::Two..=Rank::Ace, Action::Stand),
    //     (Score::Hard(20), Rank::Two..=Rank::Ace, Action::Stand),
    // ];

    // println!("---");
    // let splits = evaluate_accuracy("Split", &splits, true, &model, &device);
    // println!("---");
    // let softs = evaluate_accuracy("Soft", &softs, false, &model, &device);
    // println!("---");
    // let hards = evaluate_accuracy("Hard", &hards, false, &model, &device);

    // println!("---");
    // println!(
    //     "Total games: {}, Total win/loss: {}, EV: {:.3} %",
    //     rounds,
    //     total_win_loss,
    //     total_win_loss as f32 * 100.0 / rounds as f32
    // );
    // println!(
    //     "Hit {}, Stand {}, Double {}, Surrender {}, Split {}",
    //     action_counts[usize::from(Action::Hit)],
    //     action_counts[usize::from(Action::Stand)],
    //     action_counts[usize::from(Action::Double)],
    //     action_counts[usize::from(Action::Surrender)],
    //     action_counts[usize::from(Action::Split)]
    // );
    // println!(
    //     "Splits: {} / {} -> {:.1} %",
    //     splits.0,
    //     splits.0 + splits.1,
    //     splits.2 * 100.0
    // );
    // println!(
    //     "Softs: {} / {} -> {:.1} %",
    //     softs.0,
    //     softs.0 + softs.1,
    //     softs.2 * 100.0
    // );
    // println!(
    //     "Hards: {} / {} -> {:.1} %",
    //     hards.0,
    //     hards.0 + hards.1,
    //     hards.2 * 100.0
    // );
}
