use std::{
    sync::mpsc::{self, Receiver, Sender},
    thread::JoinHandle,
};

use burn::train::{
    TrainingInterrupter,
    metric::{MetricEntry, MetricMetadata},
    renderer::{MetricState, MetricsRenderer, TrainingProgress, tui::TuiMetricsRenderer},
};

pub struct Update {
    pub metadata: MetricMetadata,
    pub updates: Vec<(MetricEntry, f64)>,
}

pub struct Render {
    tx: Sender<(bool, Update)>,
    thread: JoinHandle<()>,
}

impl Render {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();

        let interuptor = TrainingInterrupter::new();
        let renderer = TuiMetricsRenderer::new(interuptor, None);

        let thread = std::thread::spawn(move || Self::run(renderer, rx));

        Self { tx, thread }
    }

    pub fn update_train(&self, update: Update) {
        self.tx.send((true, update)).unwrap();
    }

    pub fn update_valid(&self, update: Update) {
        self.tx.send((false, update)).unwrap();
    }

    fn run(mut renderer: TuiMetricsRenderer, rx: Receiver<(bool, Update)>) {
        rx.iter().for_each(|(train, update)| match train {
            true => {
                for update in update.updates {
                    renderer.update_train(MetricState::Numeric(update.0, update.1));
                }

                renderer.render_train(TrainingProgress {
                    progress: update.metadata.progress,
                    epoch: update.metadata.epoch,
                    epoch_total: update.metadata.epoch_total,
                    iteration: update.metadata.iteration,
                });
            }
            false => {
                for update in update.updates {
                    renderer.update_valid(MetricState::Numeric(update.0, update.1));
                }

                renderer.render_valid(TrainingProgress {
                    progress: update.metadata.progress,
                    epoch: update.metadata.epoch,
                    epoch_total: update.metadata.epoch_total,
                    iteration: update.metadata.iteration,
                });
            }
        });

        renderer.persistent();
    }

    pub fn join(self) {
        drop(self.tx);
        self.thread.join().unwrap();
    }
}
