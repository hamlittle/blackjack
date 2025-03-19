use blackjack::{
    game::game::Game,
    training::ppo::{Model, State},
};
use burn::{data::dataloader::batcher::Batcher, prelude::*};

#[derive(Clone, Debug)]
pub struct Batch<B: Backend> {
    pub games: Vec<Game>,
    pub state: Tensor<B, 2>,
}

#[derive(Clone, Debug)]
pub struct ModelBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> ModelBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<Game, Batch<B>> for ModelBatcher<B> {
    fn batch(&self, items: Vec<Game>) -> Batch<B> {
        let state: Vec<State> = items
            .iter()
            .map(|game| State {
                player_score: game.player_score(0),
                dealer_upcard: game.dealer_upcard(),
            })
            .collect();

        Batch {
            games: items,
            state: Model::<B>::normalize(&state, &self.device),
        }
    }
}
