use std::ops::RangeInclusive;

use blackjack::{
    game::{
        card::Rank,
        game::{Game, Score},
    },
    training::simulation::Action,
};

pub struct SplitRule {
    pub player: Rank,
    pub dealer: RangeInclusive<Rank>,
}

pub const SPLIT_TABLE: [SplitRule; 9] = [
    SplitRule {
        player: Rank::Two,
        dealer: Rank::Two..=Rank::Seven,
    },
    SplitRule {
        player: Rank::Three,
        dealer: Rank::Two..=Rank::Seven,
    },
    SplitRule {
        player: Rank::Four,
        dealer: Rank::Five..=Rank::Six,
    },
    SplitRule {
        player: Rank::Six,
        dealer: Rank::Two..=Rank::Six,
    },
    SplitRule {
        player: Rank::Seven,
        dealer: Rank::Two..=Rank::Seven,
    },
    SplitRule {
        player: Rank::Eight,
        dealer: Rank::Two..=Rank::Ten,
    },
    SplitRule {
        player: Rank::Nine,
        dealer: Rank::Two..=Rank::Six,
    },
    SplitRule {
        player: Rank::Nine,
        dealer: Rank::Eight..=Rank::Nine,
    },
    SplitRule {
        player: Rank::Ace,
        dealer: Rank::Two..=Rank::Ace,
    },
];

pub struct Rule {
    pub player: Score,
    pub dealer: RangeInclusive<Rank>,
    pub action: Action,
    pub alt: Option<Action>,
}

pub const SOFT_TABLE: [Rule; 23] = [
    Rule {
        player: Score::Soft(13),
        dealer: Rank::Two..=Rank::Four,
        action: Action::Hit,
        alt: None,
    },
    Rule {
        player: Score::Soft(13),
        dealer: Rank::Five..=Rank::Six,
        action: Action::Double,
        alt: Some(Action::Hit),
    },
    Rule {
        player: Score::Soft(13),
        dealer: Rank::Seven..=Rank::Ace,
        action: Action::Hit,
        alt: None,
    },
    Rule {
        player: Score::Soft(14),
        dealer: Rank::Two..=Rank::Four,
        action: Action::Hit,
        alt: None,
    },
    Rule {
        player: Score::Soft(14),
        dealer: Rank::Five..=Rank::Six,
        action: Action::Double,
        alt: Some(Action::Hit),
    },
    Rule {
        player: Score::Soft(14),
        dealer: Rank::Seven..=Rank::Ace,
        action: Action::Hit,
        alt: None,
    },
    Rule {
        player: Score::Soft(15),
        dealer: Rank::Two..=Rank::Three,
        action: Action::Hit,
        alt: None,
    },
    Rule {
        player: Score::Soft(15),
        dealer: Rank::Four..=Rank::Six,
        action: Action::Double,
        alt: Some(Action::Hit),
    },
    Rule {
        player: Score::Soft(15),
        dealer: Rank::Seven..=Rank::Ace,
        action: Action::Hit,
        alt: None,
    },
    Rule {
        player: Score::Soft(16),
        dealer: Rank::Two..=Rank::Three,
        action: Action::Hit,
        alt: None,
    },
    Rule {
        player: Score::Soft(16),
        dealer: Rank::Four..=Rank::Six,
        action: Action::Double,
        alt: Some(Action::Hit),
    },
    Rule {
        player: Score::Soft(16),
        dealer: Rank::Seven..=Rank::Ace,
        action: Action::Hit,
        alt: None,
    },
    Rule {
        player: Score::Soft(17),
        dealer: Rank::Two..=Rank::Two,
        action: Action::Hit,
        alt: None,
    },
    Rule {
        player: Score::Soft(17),
        dealer: Rank::Three..=Rank::Six,
        action: Action::Double,
        alt: Some(Action::Hit),
    },
    Rule {
        player: Score::Soft(17),
        dealer: Rank::Seven..=Rank::Ace,
        action: Action::Hit,
        alt: None,
    },
    Rule {
        player: Score::Soft(18),
        dealer: Rank::Two..=Rank::Six,
        action: Action::Double,
        alt: Some(Action::Stand),
    },
    Rule {
        player: Score::Soft(18),
        dealer: Rank::Seven..=Rank::Eight,
        action: Action::Stand,
        alt: None,
    },
    Rule {
        player: Score::Soft(18),
        dealer: Rank::Nine..=Rank::Ace,
        action: Action::Hit,
        alt: None,
    },
    Rule {
        player: Score::Soft(19),
        dealer: Rank::Two..=Rank::Five,
        action: Action::Stand,
        alt: None,
    },
    Rule {
        player: Score::Soft(19),
        dealer: Rank::Six..=Rank::Six,
        action: Action::Double,
        alt: Some(Action::Stand),
    },
    Rule {
        player: Score::Soft(19),
        dealer: Rank::Seven..=Rank::Ace,
        action: Action::Stand,
        alt: None,
    },
    Rule {
        player: Score::Soft(20),
        dealer: Rank::Two..=Rank::Ace,
        action: Action::Stand,
        alt: None,
    },
    Rule {
        player: Score::Soft(21),
        dealer: Rank::Two..=Rank::Ace,
        action: Action::Stand,
        alt: None,
    },
];

pub const HARD_TABLE: [Rule; 30] = [
    Rule {
        player: Score::Hard(4),
        dealer: Rank::Two..=Rank::Ace,
        action: Action::Hit,
        alt: None,
    },
    Rule {
        player: Score::Hard(5),
        dealer: Rank::Two..=Rank::Ace,
        action: Action::Hit,
        alt: None,
    },
    Rule {
        player: Score::Hard(6),
        dealer: Rank::Two..=Rank::Ace,
        action: Action::Hit,
        alt: None,
    },
    Rule {
        player: Score::Hard(7),
        dealer: Rank::Two..=Rank::Ace,
        action: Action::Hit,
        alt: None,
    },
    Rule {
        player: Score::Hard(8),
        dealer: Rank::Two..=Rank::Ace,
        action: Action::Hit,
        alt: None,
    },
    Rule {
        player: Score::Hard(9),
        dealer: Rank::Two..=Rank::Two,
        action: Action::Hit,
        alt: None,
    },
    Rule {
        player: Score::Hard(9),
        dealer: Rank::Three..=Rank::Six,
        action: Action::Double,
        alt: Some(Action::Hit),
    },
    Rule {
        player: Score::Hard(9),
        dealer: Rank::Seven..=Rank::Ace,
        action: Action::Hit,
        alt: None,
    },
    Rule {
        player: Score::Hard(10),
        dealer: Rank::Two..=Rank::Nine,
        action: Action::Double,
        alt: Some(Action::Hit),
    },
    Rule {
        player: Score::Hard(10),
        dealer: Rank::Ten..=Rank::Ace,
        action: Action::Hit,
        alt: None,
    },
    Rule {
        player: Score::Hard(11),
        dealer: Rank::Two..=Rank::Ace,
        action: Action::Double,
        alt: Some(Action::Hit),
    },
    Rule {
        player: Score::Hard(12),
        dealer: Rank::Two..=Rank::Three,
        action: Action::Hit,
        alt: None,
    },
    Rule {
        player: Score::Hard(12),
        dealer: Rank::Four..=Rank::Six,
        action: Action::Stand,
        alt: None,
    },
    Rule {
        player: Score::Hard(12),
        dealer: Rank::Seven..=Rank::Ace,
        action: Action::Hit,
        alt: None,
    },
    Rule {
        player: Score::Hard(13),
        dealer: Rank::Two..=Rank::Six,
        action: Action::Stand,
        alt: None,
    },
    Rule {
        player: Score::Hard(13),
        dealer: Rank::Seven..=Rank::Ace,
        action: Action::Hit,
        alt: None,
    },
    Rule {
        player: Score::Hard(14),
        dealer: Rank::Two..=Rank::Six,
        action: Action::Stand,
        alt: None,
    },
    Rule {
        player: Score::Hard(14),
        dealer: Rank::Seven..=Rank::Ace,
        action: Action::Hit,
        alt: None,
    },
    Rule {
        player: Score::Hard(15),
        dealer: Rank::Two..=Rank::Six,
        action: Action::Stand,
        alt: None,
    },
    Rule {
        player: Score::Hard(15),
        dealer: Rank::Seven..=Rank::Nine,
        action: Action::Hit,
        alt: None,
    },
    Rule {
        player: Score::Hard(15),
        dealer: Rank::Ten..=Rank::Ace,
        action: Action::Surrender,
        alt: Some(Action::Hit),
    },
    Rule {
        player: Score::Hard(16),
        dealer: Rank::Two..=Rank::Six,
        action: Action::Stand,
        alt: None,
    },
    Rule {
        player: Score::Hard(16),
        dealer: Rank::Seven..=Rank::Eight,
        action: Action::Hit,
        alt: None,
    },
    Rule {
        player: Score::Hard(16),
        dealer: Rank::Nine..=Rank::Ace,
        action: Action::Surrender,
        alt: Some(Action::Hit),
    },
    Rule {
        player: Score::Hard(17),
        dealer: Rank::Two..=Rank::King,
        action: Action::Stand,
        alt: None,
    },
    Rule {
        player: Score::Hard(17),
        dealer: Rank::Ace..=Rank::Ace,
        action: Action::Surrender,
        alt: Some(Action::Stand),
    },
    Rule {
        player: Score::Hard(18),
        dealer: Rank::Two..=Rank::Ace,
        action: Action::Stand,
        alt: None,
    },
    Rule {
        player: Score::Hard(19),
        dealer: Rank::Two..=Rank::Ace,
        action: Action::Stand,
        alt: None,
    },
    Rule {
        player: Score::Hard(20),
        dealer: Rank::Two..=Rank::Ace,
        action: Action::Stand,
        alt: None,
    },
    Rule {
        player: Score::Hard(21),
        dealer: Rank::Two..=Rank::Ace,
        action: Action::Stand,
        alt: None,
    },
];

pub fn basic_strategy(game: &Game, player: usize, filter: Option<&[Action]>) -> Option<Action> {
    let filter = match filter {
        Some(filter) => filter,
        None => &[
            Action::Hit,
            Action::Stand,
            Action::Double,
            Action::Surrender,
            Action::Split,
        ],
    };

    let prio = [
        should_split(game, player),
        should_double(game, player, true),
        should_surrender(game, player, true),
        should_hit(game, player, true),
        should_stand(game, player, true),
    ];

    let action = prio
        .into_iter()
        .flatten()
        .filter(|action| filter.contains(action))
        .next();

    if action.is_none() {
        let prio = [
            should_double(game, player, false),
            should_surrender(game, player, false),
            should_hit(game, player, false),
            should_stand(game, player, false),
        ];

        prio.into_iter()
            .flatten()
            .filter(|action| filter.contains(action))
            .next()
    } else {
        action
    }
}

pub fn should_split(game: &Game, player: usize) -> Option<Action> {
    let player_hand = game.player_hand(player);
    let dealer_upcard = game.dealer_upcard();

    if player_hand[0].rank != player_hand[1].rank {
        return None;
    }

    let player = player_hand[0].rank;
    let dealer = dealer_upcard.rank;

    for rule in &SPLIT_TABLE {
        if rule.player == player && rule.dealer.contains(&dealer) {
            return Some(Action::Split);
        }
    }

    None
}

pub fn should_double(game: &Game, player: usize, main_action: bool) -> Option<Action> {
    let player = game.player_score(player);
    let dealer = game.dealer_upcard().rank;

    if rule_lookup(player, dealer, main_action) == Some(Action::Double) {
        Some(Action::Double)
    } else {
        None
    }
}

pub fn should_surrender(game: &Game, player: usize, main_action: bool) -> Option<Action> {
    let player = game.player_score(player);
    let dealer = game.dealer_upcard().rank;

    if rule_lookup(player, dealer, main_action) == Some(Action::Surrender) {
        Some(Action::Surrender)
    } else {
        None
    }
}

pub fn should_hit(game: &Game, player: usize, main_action: bool) -> Option<Action> {
    let player = game.player_score(player);
    let dealer = game.dealer_upcard().rank;

    if rule_lookup(player, dealer, main_action) == Some(Action::Hit) {
        Some(Action::Hit)
    } else {
        None
    }
}

pub fn should_stand(game: &Game, player: usize, main_action: bool) -> Option<Action> {
    let player = game.player_score(player);
    let dealer = game.dealer_upcard().rank;

    if rule_lookup(player, dealer, main_action) == Some(Action::Stand) {
        Some(Action::Stand)
    } else {
        None
    }
}

fn rule_lookup(player: Score, dealer: Rank, main_action: bool) -> Option<Action> {
    let table = match player {
        Score::Hard(_) => &mut HARD_TABLE.iter(),
        Score::Soft(_) => &mut SOFT_TABLE.iter(),
    };

    for rule in table {
        if rule.player != player || !rule.dealer.contains(&dealer) {
            continue;
        }

        if main_action {
            return Some(rule.action);
        } else {
            return rule.alt;
        };
    }

    None
}
