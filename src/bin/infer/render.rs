use std::{
    sync::mpsc::{self, Receiver, Sender},
    thread::JoinHandle,
};

use burn::{
    prelude::*,
    train::{
        TrainingInterrupter,
        metric::{MetricEntry, MetricMetadata},
        renderer::{MetricState, MetricsRenderer, TrainingProgress, tui::TuiMetricsRenderer},
    },
};

use crate::metrics::{Metrics, Update};

pub struct Render<B: Backend> {
    metrics: Metrics<B>,
    tx: Sender<(bool, MetricMetadata, Vec<(MetricEntry, f64)>)>,
    thread: JoinHandle<()>,
}

impl<B: Backend> Render<B> {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();

        let interuptor = TrainingInterrupter::new();
        let renderer = TuiMetricsRenderer::new(interuptor, None);

        let thread = std::thread::spawn(move || Self::run(renderer, rx));

        Self {
            metrics: Metrics::new(),
            tx,
            thread,
        }
    }

    pub fn reset(&mut self) {
        self.metrics = Metrics::<B>::new();
    }

    pub fn update(&mut self, data: Update<B>, train: bool) {
        let metadata = MetricMetadata {
            progress: data.metadata.progress.clone(),
            epoch: data.metadata.epoch,
            epoch_total: data.metadata.epoch_total,
            iteration: data.metadata.iteration,
            lr: data.metadata.lr,
        };

        self.tx
            .send((train, metadata, self.metrics.update(data)))
            .unwrap();
    }

    fn run(
        mut renderer: TuiMetricsRenderer,
        rx: Receiver<(bool, MetricMetadata, Vec<(MetricEntry, f64)>)>,
    ) {
        rx.iter()
            .for_each(|(train, metadata, updates)| match train {
                true => {
                    for update in updates {
                        renderer.update_train(MetricState::Numeric(update.0, update.1));
                    }

                    renderer.render_train(TrainingProgress {
                        progress: metadata.progress,
                        epoch: metadata.epoch,
                        epoch_total: metadata.epoch_total,
                        iteration: metadata.iteration,
                    });
                }
                false => {
                    for update in updates {
                        renderer.update_valid(MetricState::Numeric(update.0, update.1));
                    }

                    renderer.render_valid(TrainingProgress {
                        progress: metadata.progress,
                        epoch: metadata.epoch,
                        epoch_total: metadata.epoch_total,
                        iteration: metadata.iteration,
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
