// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use crate::base_types::{ObjectID, SequenceNumber, SuiAddress};
use crate::collection_types::{Table, VecSet};
use crate::dynamic_field::get_dynamic_field_from_store;
use crate::error::{SuiResult, UserInputError, UserInputResult};
use crate::id::{ID, UID};
use crate::object::Owner;
use crate::storage::ObjectStore;
use move_core_types::account_address::AccountAddress;
use move_core_types::ident_str;
use move_core_types::identifier::IdentStr;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use tracing::debug;

pub const COIN_DENY_LIST_OBJECT_ID: ObjectID = ObjectID::from_address(coin_deny_list_addr());

pub const COIN_DENY_LIST_MODULE: &IdentStr = ident_str!("coin");
pub const COIN_DENY_LIST_CREATE_FUNC: &IdentStr = ident_str!("create_deny_list_object");

/// Returns 0x404
const fn coin_deny_list_addr() -> AccountAddress {
    let mut addr = [0u8; AccountAddress::LENGTH];
    addr[AccountAddress::LENGTH - 2] = 0x4;
    addr[AccountAddress::LENGTH - 1] = 0x4;
    AccountAddress::new(addr)
}

pub fn get_coin_deny_list_obj_initial_shared_version(
    object_store: &dyn ObjectStore,
) -> SuiResult<Option<SequenceNumber>> {
    Ok(object_store
        .get_object(&COIN_DENY_LIST_OBJECT_ID)?
        .map(|obj| match obj.owner {
            Owner::Shared {
                initial_shared_version,
            } => initial_shared_version,
            _ => unreachable!("Randomness state object must be shared"),
        }))
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CoinDenyList {
    pub id: UID,
    // Table<address, u64>
    pub frozen_count: Table,
    // Table<ID, VecSet<address>>
    pub frozen_addresses: Table,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FreezeCap {
    pub id: UID,
    pub package: ID,
}

impl CoinDenyList {
    pub fn check_deny_list(
        address: SuiAddress,
        coin_type_package_ids: BTreeSet<ObjectID>,
        object_store: &dyn ObjectStore,
    ) -> UserInputResult {
        let deny_list_object = match object_store.get_object(&COIN_DENY_LIST_OBJECT_ID) {
            Ok(Some(obj)) => obj,
            _ => {
                return Ok(());
            }
        };
        // Unwrap safe because the deny list object is created by the system.
        let deny_list: CoinDenyList = deny_list_object.to_rust().unwrap();
        // TODO: Add caches to avoid repeated DF reads.
        let Ok(count) = get_dynamic_field_from_store::<SuiAddress, u64>(
            object_store,
            deny_list.frozen_count.id,
            &address,
        ) else {
            return Ok(());
        };
        if count == 0 {
            return Ok(());
        }
        for coin_package_id in coin_type_package_ids {
            let Ok(denied_addresses) = get_dynamic_field_from_store::<ID, VecSet<SuiAddress>>(
                object_store,
                deny_list.frozen_addresses.id,
                &ID::new(coin_package_id),
            ) else {
                continue;
            };
            let denied_addresses: BTreeSet<_> = denied_addresses.contents.into_iter().collect();
            if denied_addresses.contains(&address) {
                debug!(
                    "Address {} is denied for coin package {}",
                    address, coin_package_id
                );
                return Err(UserInputError::AddressDeniedForCoin {
                    address,
                    coin_package_id,
                });
            }
        }
        Ok(())
    }
}
