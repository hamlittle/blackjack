use num_enum::{IntoPrimitive, TryFromPrimitive};
use strum::EnumIter;

use crate::game::game::Game;

#[derive(TryFromPrimitive, IntoPrimitive, EnumIter, Clone, Copy, PartialEq, Debug)]
#[repr(usize)]
pub enum Action {
    Hit,
    Stand,
    Double,
    Surrender,
    Split,
}

pub struct Simulation {
    game: Game,
}

impl Simulation {
    pub fn new(game: Game) -> Self {
        Self { game }
    }

    pub fn forward(mut self, player: usize, action: Action) -> Game {
        match action {
            Action::Hit => {
                self.game.player_hit(player);
            }
            Action::Stand => {
                self.game.player_stand(player);
            }
            Action::Double => {
                self.game.player_double(player);
            }
            Action::Surrender => {
                self.game.player_surrender(player);
            }
            Action::Split => {
                self.game.player_split(player);
            }
        }

        self.game
    }
}
