use std::fs;

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Data {
    pub shoe_size: usize,
    pub shoe_count: usize,
    pub no_blackjack: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Generate {
    pub workers: usize,
    pub out_dir: String,
    pub progress: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub data: Data,
    pub generate: Generate,
}

pub fn load(filename: &str) -> Result<Config> {
    let contents = fs::read_to_string(filename)?;
    let config: Config = toml::from_str(&contents)?;

    Ok(config)
}
