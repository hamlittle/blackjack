use std::time::{Duration, Instant};

use blackjack::{
    game::game::Outcome,
    model::{
        Model, Train, Valid,
        dqn::{Dqn, Learner, LearnerConfig, State, Weights},
    },
    training::{
        data::{GameDataset, ReplayBuffer, ReplayItem},
        render::{Render, Update},
        simulation::{Action, Simulation},
    },
};
use burn::{
    config::Config,
    data::{dataloader::Progress, dataset::DatasetIterator},
    module::{AutodiffModule, Module},
    optim::{Adam, adaptor::OptimizerAdaptor},
    prelude::Backend,
    record::CompactRecorder,
    tensor::{Int, Tensor, backend::AutodiffBackend, cast::ToElement},
    train::metric::{
        IterationSpeedMetric, LossInput, LossMetric, Metric, MetricMetadata, Numeric,
        state::{FormatOptions, NumericMetricState},
    },
};
use rand::{Rng, seq::IndexedRandom};

#[derive(Config)]
pub struct Hyper {
    pub weights: Weights,

    #[config(default = 10_000)]
    pub iterations: usize,

    #[config(default = 100_000)]
    pub replay: usize,

    #[config(default = 1)]
    pub batch_size: usize,

    #[config(default = 4)]
    pub gen_steps: usize,

    #[config(default = 1)]
    pub train_steps: usize,

    #[config(default = 20)]
    pub trunc: usize,
}

#[derive(Config)]
pub struct TrainingConfig {
    pub artifact_dir: String,
    pub hyper: Vec<Hyper>,
    pub learner: LearnerConfig,

    #[config(default = 0x12C0FFEE)]
    pub seed: u64,
}

impl TrainingConfig {
    pub fn init<B: AutodiffBackend>(self, device: B::Device) -> Trainer<B> {
        Trainer::new(self, device)
    }
}

pub struct Trainer<B>
where
    B: AutodiffBackend,
{
    artifact_dir: String,
    hyper: Vec<Hyper>,
    learner: Learner<B, OptimizerAdaptor<Adam, Dqn<B>, B>>,
    render: Render,
    device: B::Device,
}

impl<B> Trainer<B>
where
    B: AutodiffBackend,
{
    pub fn new(config: TrainingConfig, device: B::Device) -> Self {
        std::fs::remove_dir_all(&config.artifact_dir).ok();
        std::fs::create_dir_all(&config.artifact_dir).ok();

        config
            .save(format!("{}/train.json", config.artifact_dir))
            .expect("Config should be saved successfully");

        config
            .learner
            .model
            .save(format!("{}/model.json", config.artifact_dir))
            .expect("Config should be saved successfully");

        Self {
            artifact_dir: config.artifact_dir,
            hyper: config.hyper,
            learner: config.learner.init(device.clone()),
            render: Render::new(),
            device,
        }
    }

    pub fn run(mut self) {
        let epochs = self.hyper.clone();

        for (epoch, hyper) in epochs.iter().enumerate() {
            // train step
            let mut metrics = Metrics::<B>::new();

            let mut replay_buffer =
                ReplayBuffer::new(hyper.replay, hyper.batch_size, hyper.iterations);

            let items = self.generate(hyper.replay, hyper.trunc, 1.0);
            items.into_iter().for_each(|item| replay_buffer.push(item));

            let mut start = Instant::now();
            for iteration in 0..hyper.iterations {
                let items = self.generate(hyper.gen_steps, hyper.trunc, hyper.weights.eps);
                items.into_iter().for_each(|item| replay_buffer.push(item));

                for _ in 0..hyper.train_steps {
                    let item = self
                        .learner
                        .train_step(&replay_buffer, &hyper.weights, iteration);
                    self.learner = self.learner.fit(item.grads, &hyper.weights);

                    self.render_update(
                        &mut metrics,
                        epoch + 1,
                        epochs.len(),
                        iteration + 1,
                        start.elapsed(),
                        &hyper,
                        &item.item,
                        true,
                    );
                    start = Instant::now();
                }
            }

            self.save_model(&format!("-{}", epoch + 1));

            // valid step
            let mut metrics = Metrics::<B>::new();

            let iterations = hyper.iterations / 10;
            let weights = Weights {
                lr: 0.0,
                gamma: 1.0,
                eps: 0.0,
            };

            let mut replay_buffer = ReplayBuffer::new(hyper.replay, hyper.batch_size, iterations);

            let items = self.generate(hyper.replay, hyper.trunc, weights.eps);
            items.into_iter().for_each(|item| replay_buffer.push(item));

            let mut start = Instant::now();
            for iteration in 0..iterations {
                let items = self.generate(hyper.gen_steps, hyper.trunc, weights.eps);
                items.into_iter().for_each(|item| replay_buffer.push(item));

                for _ in 0..hyper.train_steps {
                    let item = self.learner.valid_step(&replay_buffer, &weights, iteration);

                    self.render_update(
                        &mut metrics,
                        epoch + 1,
                        epochs.len(),
                        iteration + 1,
                        start.elapsed(),
                        &hyper,
                        &item.item,
                        false,
                    );
                    start = Instant::now();
                }
            }
        }

        self.save_model("");
        self.render.join();
    }

    fn save_model(&self, suffix: &str) {
        self.learner
            .model()
            .clone()
            .save_file(
                format!("{}/model{}", self.artifact_dir, suffix),
                &CompactRecorder::new(),
            )
            .expect("Failed to save trained model");
    }

    fn generate(&self, count: usize, truncate: usize, exploration: f32) -> Vec<ReplayItem<B>> {
        let dataset = GameDataset::new(count, 6, true, Some(truncate));
        let mut iter = DatasetIterator::new(&dataset);

        (0..count)
            .into_iter()
            .map(|_| {
                let mut game = iter.next().unwrap();
                let player = game.add_player(1.0);
                game.start();

                let player_hand = game.player_hand(player);
                let can_split = player_hand[0].rank == player_hand[1].rank;
                let mask: Vec<_> = match can_split {
                    true => vec![
                        Action::Hit,
                        Action::Stand,
                        Action::Double,
                        Action::Surrender,
                        Action::Split,
                    ],
                    false => vec![
                        Action::Hit,
                        Action::Stand,
                        Action::Double,
                        Action::Surrender,
                    ],
                }
                .into_iter()
                .map(|action| action as usize)
                .collect();

                let state = State::new(game.clone(), player);
                let action = if exploration == 1.0 || rand::rng().random::<f32>() < exploration {
                    *mask.choose(&mut rand::rng()).unwrap()
                } else {
                    let state = Dqn::<B>::normalize(&[state.clone()], &self.device).valid();
                    let q = self.learner.model().valid().forward(state);

                    let select =
                        Tensor::<B, 1, Int>::from_data(mask.as_slice(), &self.device).valid();
                    q.select(1, select).argmax(1).into_scalar().to_usize()
                };

                let mut game =
                    Simulation::new(game).forward(player, Action::try_from(action).unwrap());
                let next_state = State::new(game.clone(), player);

                let terminal = !game.player_active(player);
                if terminal {
                    game.end();
                }

                let reward = match game.player_outcome(player) {
                    Some(Outcome::PlayerWin(amount)) => amount,
                    Some(Outcome::DealerWin(amount)) => -amount,
                    Some(Outcome::Push) => 0.0,
                    None => 0.0,
                };

                let split_mul = match Action::try_from(action).unwrap() {
                    Action::Split => 2.0,
                    _ => 1.0,
                };

                ReplayItem {
                    state: Tensor::<B, 2>::from_data([state.normalize()], &self.device),
                    action: Tensor::<B, 2, Int>::from_data([[action as u32]], &self.device),
                    reward: Tensor::<B, 2>::from_data([[reward]], &self.device),
                    split_mul: Tensor::<B, 2>::from_data([[split_mul]], &self.device),
                    terminal: Tensor::<B, 2>::from_data([[terminal as u8 as f32]], &self.device),
                    next_state: Tensor::<B, 2>::from_data([next_state.normalize()], &self.device),
                }
            })
            .collect()
    }

    fn render_update(
        &self,
        metrics: &mut Metrics<B>,
        epoch: usize,
        epoch_total: usize,
        iteration: usize,
        iteration_elapsed: Duration,
        hyper: &Hyper,
        loss: &Tensor<B, 1>,
        train: bool,
    ) {
        let metadata = MetricMetadata {
            progress: Progress {
                items_processed: iteration * hyper.batch_size,
                items_total: hyper.iterations * hyper.batch_size,
            },
            epoch,
            epoch_total,
            lr: Some(hyper.weights.lr as f64),
            iteration,
        };

        let mut updates = Vec::new();
        updates.push((
            metrics
                .loss
                .update(&LossInput::<B>::new(loss.clone()), &metadata),
            metrics.loss.value(),
        ));
        updates.push((
            metrics.iteration.update(&(), &metadata),
            metrics.iteration.value(),
        ));
        updates.push((
            metrics.batch.update(
                hyper.batch_size as f64 / iteration_elapsed.as_secs_f64(),
                hyper.batch_size,
                FormatOptions::new("training speed")
                    .unit("item/second")
                    .precision(0),
            ),
            metrics.batch.value(),
        ));
        updates.push((
            metrics.exploration.update(
                hyper.weights.eps as f64,
                hyper.batch_size,
                FormatOptions::new("exploration").precision(2),
            ),
            metrics.exploration.value(),
        ));
        updates.push((
            metrics.discount.update(
                hyper.weights.gamma as f64,
                hyper.batch_size,
                FormatOptions::new("discount").precision(3),
            ),
            metrics.discount.value(),
        ));
        updates.push((
            metrics.learning_rate.update(
                hyper.weights.lr as f64,
                hyper.batch_size,
                FormatOptions::new("learning rate").precision(2),
            ),
            metrics.learning_rate.value(),
        ));

        if train {
            self.render.update_train(Update { metadata, updates });
        } else {
            self.render.update_valid(Update { metadata, updates });
        }
    }
}

struct Metrics<B: Backend> {
    loss: LossMetric<B>,
    iteration: IterationSpeedMetric,
    batch: NumericMetricState,
    exploration: NumericMetricState,
    discount: NumericMetricState,
    learning_rate: NumericMetricState,
}

impl<B: Backend> Metrics<B> {
    pub fn new() -> Self {
        Self {
            loss: LossMetric::<B>::new(),
            iteration: IterationSpeedMetric::new(),
            batch: NumericMetricState::new(),
            exploration: NumericMetricState::new(),
            discount: NumericMetricState::new(),
            learning_rate: NumericMetricState::new(),
        }
    }
}
