use std::iter::zip;

use burn::{
    data::dataloader::Progress,
    module::AutodiffModule,
    nn::loss::{MseLoss, Reduction},
    optim::{GradientsParams, Optimizer},
    tensor::{Int, Tensor, TensorData, backend::AutodiffBackend, cast::ToElement},
    train::RegressionOutput,
};
use rand::Rng;

use crate::game::game::{Game, Outcome};

use super::{
    data::GameBatch,
    model::{Action, Model},
    simulation::Simulation,
};

pub struct Learner<B, O>
where
    B: AutodiffBackend,
    O: Optimizer<Model<B>, B>,
{
    online: Model<B>,
    offline: Model<B::InnerBackend>,
    optim: O,
    device: B::Device,
    offline_count: usize,
}

#[derive(Copy, Clone)]
pub struct Weights {
    pub learning_rate: f64,
    pub gamma: f32,
    pub eps: f64,
    pub offline_count: u32,
}

impl<B, O> Learner<B, O>
where
    B: AutodiffBackend,
    O: Optimizer<Model<B>, B>,
{
    pub fn new(model: Model<B>, optim: O, device: B::Device) -> Self {
        Self {
            online: model.clone(),
            offline: model.clone().valid(),
            optim,
            device,
            offline_count: 0,
        }
    }

    // pub fn valid_step(&self, batch: &GameBatch<B>, weights: &Weights) -> RegressionOutput<B> {
    //     let mut weights = weights.clone();
    //     weights.eps = 0.0;

    //     self.train_step(batch, &weights)
    // }

    pub fn train_step(&self, batch: &GameBatch<B>, weights: &Weights) -> RegressionOutput<B> {
        let q = self.online.forward(batch.input.clone());
        let actions = self.apply_exploration(q.clone(), weights.eps);
        let rewards = self.compute_next_reward(batch.games.clone(), &actions, weights.gamma);
        let target_q = self.update_target(q.clone(), &actions, &rewards);
        let loss = MseLoss::new().forward(q.clone(), target_q.clone(), Reduction::Mean);

        RegressionOutput {
            loss,
            output: q,
            targets: target_q,
        }
    }

    pub fn valid_step(&self, batch: &GameBatch<B>, _weights: &Weights) -> RegressionOutput<B> {
        let q = self.online.forward(batch.input.clone());
        let actions = self.apply_exploration(q.clone(), 0.0);
        let rewards = self.compute_terminal_reward(batch.games.clone());
        let target_q = self.update_target(q.clone(), &actions, &rewards);
        let loss = MseLoss::new().forward(q.clone(), target_q.clone(), Reduction::Mean);

        RegressionOutput {
            loss,
            output: q,
            targets: target_q,
        }
    }

    pub fn optim(
        mut self,
        regression: &RegressionOutput<B>,
        weights: &Weights,
        progress: &Progress,
    ) -> Self {
        let grads = GradientsParams::from_grads(regression.loss.backward(), &self.online);
        self.online = self.optim.step(weights.learning_rate, self.online, grads);

        if progress.items_processed - self.offline_count > weights.offline_count as usize {
            self.offline = self.online.valid().clone();
            self.offline_count = progress.items_processed;
        }

        self
    }

    pub fn model(&self) -> Model<B> {
        self.online.clone()
    }

    fn apply_exploration(&self, decisions: Tensor<B, 2>, exploration: f64) -> Vec<Action> {
        let best_decision = decisions.argmax(1).to_data();

        best_decision
            .iter()
            .map(|action: B::IntElem| {
                if rand::rng().random::<f64>() < exploration {
                    rand::rng().random_range(0..Model::<B>::output_size())
                } else {
                    action.to_usize()
                }
            })
            .map(|action| Action::try_from(action).unwrap())
            .collect()
    }

    fn compute_next_reward(&self, games: Vec<Game>, actions: &[Action], discount: f32) -> Vec<f32> {
        assert_eq!(games.len(), actions.len());

        zip(games, actions)
            .into_iter()
            .map(|(game, action)| Simulation::new(game).forward(*action))
            .map(|game| match game.player_outcome(0) {
                Some(outcome) => match outcome {
                    Outcome::PlayerWin(amount) => amount,
                    Outcome::Push => 0.0,
                    Outcome::DealerWin(bet) => -bet,
                },
                None => {
                    let new_state =
                        Model::normalize(game.player_score(0), game.dealer_upcard(), &self.device);
                    let next_q = self.offline.forward(new_state.clone());
                    let predicted_reward = next_q.max_dim(1).into_scalar().to_f32();

                    discount * predicted_reward
                }
            })
            .collect()
    }

    fn compute_terminal_reward(&self, games: Vec<Game>) -> Vec<f32> {
        games
            .into_iter()
            .map(|mut game| {
                let mut first_hand = true;
                while game.player_outcome(0).is_none() {
                    let input = Model::<B>::normalize(
                        game.player_score(0),
                        game.dealer_upcard(),
                        &self.device,
                    );
                    let q = self.model().valid().forward(input.valid());

                    let valid_q = if !first_hand {
                        let filter = [Action::Hit, Action::Stand].map(usize::from);
                        let filter = Tensor::<B, 1, Int>::from_data(filter, &self.device);

                        q.select(1, filter.valid())
                    } else {
                        q
                    };

                    let action: Action = valid_q
                        .argmax(1)
                        .into_scalar()
                        .to_usize()
                        .try_into()
                        .unwrap();

                    first_hand = false;
                    game = Simulation::new(game).forward(action);
                }

                match game.player_outcome(0).unwrap() {
                    Outcome::PlayerWin(amount) => amount,
                    Outcome::DealerWin(bet) => bet,
                    Outcome::Push => 0.0,
                }
            })
            .collect()
    }

    fn update_target(
        &self,
        input: Tensor<B, 2>,
        actions: &[Action],
        rewards: &[f32],
    ) -> Tensor<B, 2> {
        let dims = input.dims();

        assert_eq!(dims[0], actions.len());
        assert_eq!(dims[0], rewards.len());

        let mut data = input.to_data().to_vec::<f32>().unwrap();

        zip(actions, rewards)
            .enumerate()
            .for_each(|(batch, (action, reward))| {
                let row: usize = batch * dims[1];
                let col: usize = (*action).try_into().unwrap();
                data[row + col] = *reward;
            });

        Tensor::<B, 2>::from_data(TensorData::new(data, dims), &self.device)
    }
}
