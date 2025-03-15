use crate::game::game::{Game, Outcome};

use super::model::Action;

pub struct Simulation {
    game: Game,
}

impl Simulation {
    pub fn new(game: Game) -> Self {
        Self { game }
    }

    pub fn forward(&mut self, action: Action) -> Vec<Option<Outcome>> {
        match action {
            Action::Hit => {
                self.game.player_hit(0);
                vec![self.game.player_outcome(0)]
            }
            Action::Stand => {
                self.game.player_stand(0);
                self.game.end();
                vec![self.game.player_outcome(0)]
            } // Action::Double => {
              //     game.player_double(0);
              //     vec![0]
              // }
              // Action::Split => {
              //     let split_play = game.player_split(0);
              //     vec![0, split_play]
              // }
              // Action::Surrender => {
              //     game.player_surrender(0);
              //     vec![0]
              // }
        }
    }
}
