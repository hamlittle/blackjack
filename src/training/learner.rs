use std::iter::zip;

use burn::{
    nn::loss::{MseLoss, Reduction},
    optim::{GradientsParams, Optimizer},
    tensor::{Tensor, backend::AutodiffBackend, cast::ToElement},
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
    model: Model<B>,
    optim: O,
    device: B::Device,
}

#[derive(Copy, Clone)]
pub struct Weights {
    pub learning_rate: f64,
    pub gamma: f32,
    pub eps: f32,
}

impl<B, O> Learner<B, O>
where
    B: AutodiffBackend,
    O: Optimizer<Model<B>, B>,
{
    pub fn new(model: Model<B>, optim: O, device: B::Device) -> Self {
        Self {
            model: model,
            optim,
            device,
        }
    }

    pub fn valid_step(&self, batch: &GameBatch<B>, weights: &Weights) -> RegressionOutput<B> {
        let mut weights = weights.clone();
        weights.eps = 0.0;

        self.train_step(batch, &weights)
    }

    pub fn train_step(&self, batch: &GameBatch<B>, weights: &Weights) -> RegressionOutput<B> {
        let decisions = self.model.forward(batch.input.clone());
        let actions = self.apply_exploration(decisions.clone(), weights.eps);
        let rewards = self.compute_reward(zip(&batch.games, &actions), weights.gamma);
        let targets = self.update_target(decisions.clone(), zip(&actions, &rewards));
        let loss = MseLoss::new().forward(decisions.clone(), targets.clone(), Reduction::Mean);

        RegressionOutput {
            loss,
            output: decisions,
            targets,
        }
    }

    pub fn optim(mut self, regression: &RegressionOutput<B>, weights: &Weights) -> Self {
        let grads = GradientsParams::from_grads(regression.loss.backward(), &self.model);
        self.model = self.optim.step(weights.learning_rate, self.model, grads);

        self
    }

    pub fn model(&self) -> Model<B> {
        self.model.clone()
    }

    fn apply_exploration(&self, decisions: Tensor<B, 2>, exploration: f32) -> Vec<Action> {
        let best_decision = decisions.argmax(1).to_data();

        best_decision
            .iter()
            .map(|action: B::IntElem| {
                if rand::rng().random::<f32>() < exploration {
                    rand::rng().random_range(0..Model::<B>::output_size())
                } else {
                    action.to_usize()
                }
            })
            .map(|action| Action::try_from(action).unwrap())
            .collect()
    }

    fn compute_reward<'a, I>(&self, batch: I, discount: f32) -> Vec<f32>
    where
        I: IntoIterator<Item = (&'a Game, &'a Action)>,
    {
        batch
            .into_iter()
            .map(|(game, action)| (game, Simulation::new(game.clone()).forward(*action)))
            .map(|(game, outcomes)| {
                outcomes
                    .into_iter()
                    .map(|outcome| match outcome {
                        Some(outcome) => self.compute_current_reward(outcome),
                        None => self.compute_future_reward(game, discount),
                    })
                    .sum()
            })
            .collect()
    }

    fn compute_current_reward(&self, outcome: Outcome) -> f32 {
        match outcome {
            Outcome::PlayerWin(amount) => amount,
            Outcome::DealerWin(amount) => -amount,
            Outcome::Push => 0.0,
        }
    }

    fn compute_future_reward(&self, game: &Game, discount: f32) -> f32 {
        let state = Model::normalize(game, &self.device);

        let predicted = self.model.forward(state);
        let predicted_best = predicted
            .max_dim(1)
            .flatten::<1>(0, 1)
            .into_scalar()
            .to_f32();

        discount * predicted_best
    }

    fn update_target<'a, I>(&self, target: Tensor<B, 2>, batch: I) -> Tensor<B, 2>
    where
        I: IntoIterator<Item = (&'a Action, &'a f32)>,
    {
        batch
            .into_iter()
            .map(|(action, reward)| (usize::from(*action), reward))
            .enumerate()
            .fold(target, |target, (channel, (action, reward))| {
                target.slice_assign(
                    [channel..channel + 1, action..action + 1],
                    Tensor::<B, 2>::from_floats([[*reward]], &self.device),
                )
            })

        // let dims = target.dims();
        // let mut data = target.to_data().to_vec::<f32>().unwrap();

        // batch
        //     .into_iter()
        //     .enumerate()
        //     .for_each(|(channel, (action, reward))| {
        //         let row: usize = channel * dims[1];
        //         let col: usize = (*action).try_into().unwrap();
        //         data[row + col] = *reward;
        //     });

        // Tensor::<B, 2>::from_data(TensorData::new(data, dims), &self.device)
    }
}
