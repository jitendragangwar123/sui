// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

#[derive(Clone, Debug)]
pub(crate) enum ObjectStatus {
    OutsideConsistentReadRange,
    DeletedOrWrapped,
}
