use blackjack::config;
use blackjack::shoe::Shoe;
use num_format::{Locale, ToFormattedString};
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;
use std::{fs, thread};
use std::{io::Write, time::Instant};

fn print_progress(ndx: usize, count: usize, bytes: usize, start: Instant) {
    let elapsed = start.elapsed();
    let ratio = (ndx as f64) / count as f64;
    let data_rate = if elapsed.as_secs() > 0 {
        (bytes as f64 / 1_000_000.0) / elapsed.as_secs_f64()
    } else {
        0.0
    };
    let remaining = if ratio > 0.0 {
        elapsed.mul_f64(1.0 / ratio) - elapsed
    } else {
        std::time::Duration::ZERO
    };

    println!(
        "[{} / {}] {:.1}% - {:0.1} MB/s - remaining: {:.0?}",
        ndx.to_formatted_string(&Locale::en),
        (count).to_formatted_string(&Locale::en),
        (ndx as f64 / count as f64) * 100.0,
        data_rate,
        remaining,
    );
}

fn main() {
    let config = config::load("config.toml").unwrap();
    let gen_file = PathBuf::from(format!(
        "{}/shoes.{}-{}.ndmpk",
        config.generate.out_dir, config.data.shoe_size, config.data.shoe_count
    ));

    println!("Configuration: {:#?}", config);
    println!("Generating to file: {:?}", gen_file);

    fs::create_dir_all(gen_file.parent().unwrap()).unwrap();
    let mut gen_file = fs::File::create(gen_file).unwrap();

    let start = Instant::now();
    let (tx, rx) = mpsc::channel();

    let workers: Vec<_> = (0..config.generate.workers)
        .map(|_| {
            let count = config.data.shoe_count / config.generate.workers;
            let shoe_size = config.data.shoe_size;
            let tx = tx.clone();

            thread::spawn(move || {
                let mut rng = rand::rng();

                (0..count).for_each(|_| {
                    let mut shoe = Shoe::new(shoe_size);
                    shoe.shuffle(&mut rng);

                    let data = rmp_serde::encode::to_vec(&shoe).unwrap();

                    tx.send(data).unwrap();
                });
            })
        })
        .collect();

    drop(tx);

    let mut progress = Instant::now();
    let mut bytes: usize = 0;
    rx.iter().enumerate().for_each(|(ndx, shoe)| {
        let data = [shoe.as_slice(), b"\n"].concat();
        gen_file.write_all(&data).unwrap();

        bytes += data.len();

        if config.generate.progress > 0
            && progress.elapsed() > Duration::from_secs(config.generate.progress as u64)
        {
            print_progress(ndx, config.data.shoe_count, bytes, start);
            progress = Instant::now();
        }
    });

    let duration = start.elapsed();

    workers
        .into_iter()
        .for_each(|worker| worker.join().unwrap());

    println!("---");
    println!("Total time: {:.2?}", duration);
    println!(
        "Time per shuffle: {:.2}us",
        (duration.as_micros() as f64) / config.data.shoe_count as f64
    );
    println!(
        "Final data rate: {:.2} MB/s",
        (bytes as f64 / 1_000_000.0) / duration.as_secs_f64()
    );
    println!(
        "File size: {:.2} GB",
        gen_file.metadata().unwrap().len() as f64 / 1_000_000_000.0
    );
}
