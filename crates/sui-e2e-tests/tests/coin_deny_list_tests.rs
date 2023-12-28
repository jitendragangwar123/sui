// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use sui_core::authority::epoch_start_configuration::EpochStartConfigTrait;
use sui_json_rpc_types::SuiTransactionBlockKind;
use sui_json_rpc_types::{SuiTransactionBlockDataAPI, SuiTransactionBlockResponseOptions};
use sui_macros::sim_test;
use sui_types::coin_deny_list::{
    get_coin_deny_list_obj_initial_shared_version, CoinDenyList, COIN_DENY_LIST_OBJECT_ID,
};
use sui_types::id::UID;
use sui_types::storage::ObjectStore;
use test_cluster::TestClusterBuilder;

#[sim_test]
async fn test_coin_deny_list_creation() {
    let test_cluster = TestClusterBuilder::new()
        .with_protocol_version(32.into())
        .with_epoch_duration_ms(10000)
        .build()
        .await;
    for handle in test_cluster.all_node_handles() {
        handle.with(|node| {
            assert!(
                get_coin_deny_list_obj_initial_shared_version(&node.state().database)
                    .unwrap()
                    .is_none()
            );
            assert!(!node
                .state()
                .epoch_store_for_testing()
                .coin_deny_list_state_exists());
        });
    }
    test_cluster.wait_for_epoch_all_nodes(2).await;
    let mut prev_tx = None;
    for handle in test_cluster.all_node_handles() {
        handle.with(|node| {
            assert_eq!(
                node.state()
                    .epoch_store_for_testing()
                    .protocol_version()
                    .as_u64(),
                33
            );
            let version = node
                .state()
                .epoch_store_for_testing()
                .epoch_start_config()
                .coin_deny_list_obj_initial_shared_version()
                .unwrap();

            let deny_list_object = node
                .state()
                .database
                .get_object(&COIN_DENY_LIST_OBJECT_ID)
                .unwrap()
                .unwrap();
            assert_eq!(deny_list_object.version(), version);
            assert!(deny_list_object.owner.is_shared());
            if let Some(prev_tx) = prev_tx {
                assert_eq!(deny_list_object.previous_transaction, prev_tx);
            } else {
                prev_tx = Some(deny_list_object.previous_transaction);
            }
            let deny_list: CoinDenyList = deny_list_object.to_rust().unwrap();
            assert_eq!(deny_list.id, UID::new(COIN_DENY_LIST_OBJECT_ID));
            assert_eq!(deny_list.frozen_count.size, 0);
            assert_eq!(deny_list.frozen_addresses.size, 0);
        });
    }
    let prev_tx = prev_tx.unwrap();
    let tx = test_cluster
        .fullnode_handle
        .sui_client
        .read_api()
        .get_transaction_with_options(prev_tx, SuiTransactionBlockResponseOptions::full_content())
        .await
        .unwrap()
        .transaction
        .unwrap();
    assert!(matches!(
        tx.data.transaction(),
        SuiTransactionBlockKind::EndOfEpochTransaction(_)
    ));
    test_cluster.wait_for_epoch_all_nodes(3).await;
    // Check that we are not re-creating the same object again.
    for handle in test_cluster.all_node_handles() {
        handle.with(|node| {
            assert_eq!(
                node.state()
                    .database
                    .get_object(&COIN_DENY_LIST_OBJECT_ID)
                    .unwrap()
                    .unwrap()
                    .previous_transaction,
                prev_tx
            );
        });
    }
}
