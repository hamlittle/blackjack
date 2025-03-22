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

    let dir = "/tmp/training/";
    // let dir = "training/dqn/4";
    let model = "model-5";
    let verbose = false;
    let rounds = 500_000;

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

    println!("-- % EV ({} hands)", rounds);
    println!("Model: {:.2} %", model_ev * 100.0 / rounds as f32);
    println!("Strat: {:.2} %", strat_ev * 100.0 / rounds as f32);
    println!("-> {:.2}", (model_ev - strat_ev) * 100.0 / rounds as f32);
}
