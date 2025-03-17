use std::iter::zip;

use burn::{
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
    target: Model<B>,
    optim: O,
    device: B::Device,
    iteration: usize,
}

#[derive(Copy, Clone)]
pub struct Weights {
    pub learning_rate: f64,
    pub gamma: f32,
    pub eps: f64,
    pub target_update: u32,
}

impl<B, O> Learner<B, O>
where
    B: AutodiffBackend,
    O: Optimizer<Model<B>, B>,
{
    pub fn new(model: Model<B>, optim: O, device: B::Device) -> Self {
        Self {
            online: model.clone(),
            target: model,
            optim,
            device,
            iteration: 0,
        }
    }

    pub fn valid_step(&self, batch: &GameBatch<B>, weights: &Weights) -> RegressionOutput<B> {
        let mut weights = weights.clone();
        weights.eps = 0.0;

        self.train_step(batch, &weights)
    }

    pub fn train_step(&self, batch: &GameBatch<B>, weights: &Weights) -> RegressionOutput<B> {
        let decisions = self.online.forward(batch.input.clone());
        let actions = self.apply_exploration(decisions.clone(), weights.eps);
        let rewards = self.compute_reward(
            decisions.clone(),
            zip(&batch.games, &actions),
            weights.gamma,
        );
        let targets = self.update_target(decisions.clone(), zip(&actions, &rewards));
        let loss = MseLoss::new().forward(decisions.clone(), targets.clone(), Reduction::Mean);

        RegressionOutput {
            loss,
            output: decisions,
            targets,
        }
    }

    pub fn optim(
        mut self,
        regression: &RegressionOutput<B>,
        weights: &Weights,
        iteration: usize,
    ) -> Self {
        let grads = GradientsParams::from_grads(regression.loss.backward(), &self.online);
        self.online = self.optim.step(weights.learning_rate, self.online, grads);

        if iteration - self.iteration > weights.target_update as usize {
            self.target = self.online.clone();
            self.iteration = iteration;
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

    fn compute_reward<'a, I>(&self, decisions: Tensor<B, 2>, batch: I, discount: f32) -> Vec<f32>
    where
        I: IntoIterator<Item = (&'a Game, &'a Action)>,
        I::IntoIter: Send,
    {
        // batch
        //     .into_iter()
        //     .map(|(game, action)| (game, Simulation::new(game.clone()).forward(*action)))
        //     .map(|(game, outcomes)| {
        //         outcomes
        //             .into_iter()
        //             .map(|outcome| match outcome {
        //                 Some(outcome) => self.compute_current_reward(outcome),
        //                 None => self.compute_future_reward(game, discount),
        //             })
        //             .sum()
        //     })
        //     .collect()

        // Get the outcomes for each game (may or may not be terminal)
        let outcomes: Vec<_> = batch
            .into_iter()
            .map(|(game, action)| Simulation::new(game.clone()).forward(*action))
            .collect();

        // Store original indices to maintain order
        let (terminated, unterminated): (Vec<_>, Vec<_>) = outcomes
            .iter()
            .enumerate()
            .partition(|(_, outcome)| outcome.is_some());

        // Compute rewards for terminated hands (in order)
        let mut rewards = vec![0.0; outcomes.len()];

        for (ndx, outcome) in &terminated {
            rewards[*ndx] = match outcome.unwrap() {
                Outcome::PlayerWin(amount) => amount,
                Outcome::Push => 0.0,
                Outcome::DealerWin(bet) => -bet,
            }
        }

        // If no unterminated states exist, return immediately
        if unterminated.is_empty() {
            return rewards;
        }

        // create a tensor to select the indices of the unterminated games from the forward-pass
        let unterminated_games: Vec<_> = unterminated.iter().map(|(ndx, _)| *ndx).collect();
        let indices = Tensor::<B, 1, Int>::from_data(unterminated_games.as_slice(), &self.device);
        let future_rewards = decisions.select(0, indices).max_dim(1);

        // apply discounting for future rewards
        let discounted_rewards = future_rewards * discount;

        // insert the discounted future rewards into the empty slots
        let discounted_rewards = discounted_rewards.to_data().to_vec().unwrap();
        for (ndx, game) in unterminated_games.iter().enumerate() {
            rewards[*game] = discounted_rewards[ndx]
        }

        // sanity check: Union(terminated, unterminated) should account for every game in the batch
        let mut all_games: Vec<usize> = terminated.iter().map(|(game, _)| *game).collect();
        all_games.extend(unterminated_games);
        all_games.sort();

        let expected: Vec<usize> = (0..outcomes.len()).collect();
        assert_eq!(expected, all_games);

        // return the combined current (terminated) and future rewards (unterminated) rewards
        rewards

        // // Collect unterminated hands into a batch tensor
        // let states: Vec<Tensor<B, 2>> = unterminated
        //     .iter()
        //     .map(|(_, (game, _))| {
        //         Model::normalize(game.player_score(0), game.dealer_upcard(), &self.device)
        //     })
        //     .collect();

        // let batch_tensor = Tensor::cat(states, 0);

        // // Single forward pass for all unterminated states
        // let predicted_qs = self.target.forward(batch_tensor);

        // Extract max Q-values for each state
        // let predicted_rewards = predicted_qs.max_dim(1).to_data().to_vec::<f32>().unwrap();

        // Assign discounted rewards in the correct order
    }

    fn update_target<'a, I>(&self, target: Tensor<B, 2>, batch: I) -> Tensor<B, 2>
    where
        I: IntoIterator<Item = (&'a Action, &'a f32)>,
    {
        let dims = target.dims();
        let mut data = target.to_data().to_vec::<f32>().unwrap();

        batch
            .into_iter()
            .enumerate()
            .for_each(|(channel, (action, reward))| {
                let row: usize = channel * dims[1];
                let col: usize = (*action).try_into().unwrap();
                data[row + col] = *reward;
            });

        Tensor::<B, 2>::from_data(TensorData::new(data, dims), &self.device)
    }
}
