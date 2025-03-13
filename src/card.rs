use std::fmt::Display;

use serde::{Deserialize, Serialize};
use strum::EnumIter;

#[derive(Debug, PartialEq, Copy, Clone, EnumIter)]
pub enum Rank {
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
    Jack,
    Queen,
    King,
    Ace,
}

impl Rank {
    pub fn value(&self) -> u8 {
        use Rank::*;
        match self {
            Two => 2,
            Three => 3,
            Four => 4,
            Five => 5,
            Six => 6,
            Seven => 7,
            Eight => 8,
            Nine => 9,
            Ten | Jack | Queen | King => 10,
            Ace => 11,
        }
    }
}

#[derive(Debug, PartialEq, Copy, Clone, EnumIter)]
pub enum Suit {
    Spades,
    Hearts,
    Diamonds,
    Clubs,
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct Card {
    pub rank: Rank,
    pub suit: Suit,
}

impl Card {
    pub fn new(rank: Rank, suit: Suit) -> Self {
        Card { suit, rank }
    }
}

impl Serialize for Rank {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        use Rank::*;
        let rank = match self {
            Two => "2",
            Three => "3",
            Four => "4",
            Five => "5",
            Six => "6",
            Seven => "7",
            Eight => "8",
            Nine => "9",
            Ten => "T",
            Jack => "J",
            Queen => "Q",
            King => "K",
            Ace => "A",
        };
        serializer.serialize_str(rank)
    }
}

impl<'de> Deserialize<'de> for Rank {
    fn deserialize<D>(deserializer: D) -> Result<Rank, D::Error>
    where
        D: serde::de::Deserializer<'de>,
    {
        use serde::de::Error;
        let rank = String::deserialize(deserializer)?;
        match rank.as_str() {
            "2" => Ok(Rank::Two),
            "3" => Ok(Rank::Three),
            "4" => Ok(Rank::Four),
            "5" => Ok(Rank::Five),
            "6" => Ok(Rank::Six),
            "7" => Ok(Rank::Seven),
            "8" => Ok(Rank::Eight),
            "9" => Ok(Rank::Nine),
            "T" => Ok(Rank::Ten),
            "J" => Ok(Rank::Jack),
            "Q" => Ok(Rank::Queen),
            "K" => Ok(Rank::King),
            "A" => Ok(Rank::Ace),
            _ => Err(D::Error::custom("Invalid rank")),
        }
    }
}

impl Serialize for Suit {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        use Suit::*;
        let suit = match self {
            Spades => "S",
            Hearts => "H",
            Diamonds => "D",
            Clubs => "C",
        };
        serializer.serialize_str(suit)
    }
}

impl<'de> Deserialize<'de> for Suit {
    fn deserialize<D>(deserializer: D) -> Result<Suit, D::Error>
    where
        D: serde::de::Deserializer<'de>,
    {
        use serde::de::Error;
        let suit = String::deserialize(deserializer)?;
        match suit.as_str() {
            "S" => Ok(Suit::Spades),
            "H" => Ok(Suit::Hearts),
            "D" => Ok(Suit::Diamonds),
            "C" => Ok(Suit::Clubs),
            _ => Err(D::Error::custom("Invalid suit")),
        }
    }
}

impl Serialize for Card {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        let card = format!("{}{}", self.rank, self.suit);
        serializer.serialize_str(&card)
    }
}

impl<'de> Deserialize<'de> for Card {
    fn deserialize<D>(deserializer: D) -> Result<Card, D::Error>
    where
        D: serde::de::Deserializer<'de>,
    {
        use serde::de::Error;
        let card = String::deserialize(deserializer)?;
        let rank = match &card[0..1] {
            "2" => Rank::Two,
            "3" => Rank::Three,
            "4" => Rank::Four,
            "5" => Rank::Five,
            "6" => Rank::Six,
            "7" => Rank::Seven,
            "8" => Rank::Eight,
            "9" => Rank::Nine,
            "T" => Rank::Ten,
            "J" => Rank::Jack,
            "Q" => Rank::Queen,
            "K" => Rank::King,
            "A" => Rank::Ace,
            _ => return Err(D::Error::custom("Invalid rank")),
        };
        let suit = match &card[1..2] {
            "S" => Suit::Spades,
            "H" => Suit::Hearts,
            "D" => Suit::Diamonds,
            "C" => Suit::Clubs,
            _ => return Err(D::Error::custom("Invalid suit")),
        };
        Ok(Card::new(rank, suit))
    }
}

impl Display for Card {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.rank, self.suit)
    }
}

impl Display for Rank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Rank::*;
        let rank = match self {
            Two => "2",
            Three => "3",
            Four => "4",
            Five => "5",
            Six => "6",
            Seven => "7",
            Eight => "8",
            Nine => "9",
            Ten => "T",
            Jack => "J",
            Queen => "Q",
            King => "K",
            Ace => "A",
        };
        write!(f, "{}", rank)
    }
}

impl Display for Suit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Suit::*;
        let suit = match self {
            Spades => "S",
            Hearts => "H",
            Diamonds => "D",
            Clubs => "C",
        };
        write!(f, "{}", suit)
    }
}
