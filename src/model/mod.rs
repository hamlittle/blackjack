use burn::{optim::GradientsParams, prelude::*, train::TrainOutput};

pub mod dqn;

pub trait Model<B>
where
    B: Backend,
{
    type State;

    fn input_size() -> usize;
    fn output_size() -> usize;

    fn normalize(state: &[Self::State], device: &B::Device) -> Tensor<B, 2>;
    fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2>;
}

pub trait Train<Item> {
    type Batch;
    type Weights;

    fn train_step(
        &self,
        batch: &Self::Batch,
        weights: &Self::Weights,
        step: usize,
    ) -> TrainOutput<Item>;

    fn fit(self, grads: GradientsParams, weights: &Self::Weights, iteration: usize) -> Self;
}

pub trait Valid<Item> {
    type Batch;
    type Weights;

    fn valid_step(
        &self,
        batch: &Self::Batch,
        weights: &Self::Weights,
        step: usize,
    ) -> TrainOutput<Item>;
}
