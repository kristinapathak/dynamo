// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use thiserror::Error;

pub type ReplayResult<T> = Result<T, ReplayError>;

#[derive(Debug, Error, PartialEq)]
pub enum ReplayError {
    #[error("invalid replay spec: {0}")]
    InvalidSpec(String),

    #[error("engine error: {0}")]
    Engine(String),

    #[error("placement policy error: {0}")]
    Placement(String),

    #[error("scaling policy error: {0}")]
    Scaling(String),

    #[error("replay reached quiescence with {unfinished_requests} unfinished request(s)")]
    Deadlock { unfinished_requests: usize },

    #[error("replay invariant violated: {0}")]
    Invariant(String),
}
