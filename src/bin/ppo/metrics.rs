use std::time::Duration;

use burn::{
    prelude::*,
    train::metric::{
        IterationSpeedMetric, LossInput, LossMetric, Metric, MetricEntry, MetricMetadata, Numeric,
        state::{FormatOptions, NumericMetricState},
    },
};

pub struct Metrics<B: Backend> {
    pub win_loss: NumericMetricState,
    pub loss: LossMetric<B>,
    pub iteration: IterationSpeedMetric,
    pub batch: NumericMetricState,
    pub discont: NumericMetricState,
    pub learning_rate: NumericMetricState,
}

pub struct Update<B: Backend> {
    pub metadata: MetricMetadata,
    pub win_loss: f64,
    pub elapsed: Duration,
    pub batch_size: usize,
    pub discount: f64,
    pub learning_rate: f64,
    pub loss: Tensor<B, 1>,
}

impl<B: Backend> Metrics<B> {
    pub fn new() -> Self {
        Self {
            win_loss: NumericMetricState::new(),
            loss: LossMetric::new(),
            iteration: IterationSpeedMetric::new(),
            batch: NumericMetricState::new(),
            discont: NumericMetricState::new(),
            learning_rate: NumericMetricState::new(),
        }
    }

    pub fn update(&mut self, data: Update<B>) -> Vec<(MetricEntry, f64)> {
        let mut updates = Vec::new();

        updates.push((
            self.loss
                .update(&LossInput::<B>::new(data.loss), &data.metadata),
            self.loss.value(),
        ));
        updates.push((
            self.win_loss.update(
                data.win_loss,
                data.batch_size,
                FormatOptions::new("win/loss").unit("$").precision(2),
            ),
            self.win_loss.value(),
        ));
        updates.push((
            self.iteration.update(&(), &data.metadata),
            self.iteration.value(),
        ));
        updates.push((
            self.batch.update(
                data.batch_size as f64 / data.elapsed.as_secs_f64(),
                data.batch_size,
                FormatOptions::new("batch speed")
                    .unit("item/second")
                    .precision(0),
            ),
            self.batch.value(),
        ));
        updates.push((
            self.discont.update(
                data.discount,
                data.batch_size,
                FormatOptions::new("discount").precision(2),
            ),
            self.discont.value(),
        ));
        updates.push((
            self.learning_rate.update(
                data.learning_rate,
                data.batch_size,
                FormatOptions::new("learning rate").precision(2),
            ),
            self.learning_rate.value(),
        ));

        updates
    }
}
