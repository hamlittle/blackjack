use blackjack::game::card::Rank;

use crate::eval::AccuracyReport;

pub fn print_accuracy(header: &str, reports: &[AccuracyReport], verbose: bool) {
    println!("--- {}", header);

    let reports = reports
        .iter()
        .filter(|report| ![Rank::Jack, Rank::Queen, Rank::King].contains(&report.dealer))
        .filter(|report| verbose || report.action != report.correct);

    for report in reports {
        let symbol = if report.action == report.correct {
            "\u{2705}"
        } else {
            "\u{274C}"
        };

        println!(
            "{} {:?} vs {} | {:?} ({:?}) -> {}",
            symbol,
            report.player,
            report.dealer,
            report.action,
            report.correct,
            report.q.clone().into_data()
        );
    }
}

pub fn print_summary(header: &str, reports: &[AccuracyReport]) {
    let correct = reports
        .iter()
        .filter(|report| report.action == report.correct)
        .count();
    let total = reports.len();
    let ratio = correct as f32 * 100.0 / total as f32;

    println!("{:8}: {:.1} % ({} / {})", header, ratio, correct, total);
}
