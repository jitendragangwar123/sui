// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use sui_core::authority::epoch_start_configuration::EpochStartConfigTrait;
use sui_json_rpc_types::SuiTransactionBlockEffectsAPI;
use sui_json_rpc_types::SuiTransactionBlockKind;
use sui_json_rpc_types::{SuiTransactionBlockDataAPI, SuiTransactionBlockResponseOptions};
use sui_macros::sim_test;
use sui_types::base_types::{ObjectID, SequenceNumber, SuiAddress};
use sui_types::coin_deny_list::{
    get_coin_deny_list_obj_initial_shared_version, CoinDenyList, FreezeCap, COIN_DENY_LIST_MODULE,
    COIN_DENY_LIST_OBJECT_ID,
};
use sui_types::error::UserInputError;
use sui_types::id::UID;
use sui_types::object::Object;
use sui_types::storage::ObjectStore;
use sui_types::transaction::{CallArg, ObjectArg};
use sui_types::SUI_FRAMEWORK_PACKAGE_ID;
use test_cluster::{TestCluster, TestClusterBuilder};
use tracing::info;

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

#[sim_test]
async fn test_sender_denied_by_coin() {
    let (test_cluster, test_context) = setup_and_publish_regulated_coin().await;
    deny_address(&test_cluster, test_context.sender, &test_context).await;
    let gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(test_context.sender)
        .await
        .unwrap()
        .unwrap();
    let tx_data = test_cluster
        .test_transaction_builder_with_gas_object(test_context.sender, gas)
        .await
        .transfer(
            test_context.new_coin.compute_object_reference(),
            test_context.sender,
        )
        .build();
    let tx = test_cluster.sign_transaction(&tx_data);
    let result = test_cluster.wallet.execute_transaction_may_fail(tx).await;
    let expected_error = UserInputError::AddressDeniedForCoin {
        address: test_context.sender,
        coin_package_id: test_context.coin_package,
    };
    assert!(result
        .unwrap_err()
        .to_string()
        .contains(&expected_error.to_string()));
}

#[derive(Debug)]
struct TestContext {
    coin_deny_list_object_init_version: SequenceNumber,
    coin_package: ObjectID,
    new_coin: Object,
    deny_cap_object: Object,
    // Owner address of the new coin and deny cap object.
    sender: SuiAddress,
}

// Returns the test cluster and the deny cap.
async fn setup_and_publish_regulated_coin() -> (TestCluster, TestContext) {
    let test_cluster = TestClusterBuilder::new().build().await;
    let coin_deny_list_object_init_version = test_cluster
        .fullnode_handle
        .sui_node
        .state()
        .epoch_store_for_testing()
        .epoch_start_config()
        .coin_deny_list_obj_initial_shared_version()
        .unwrap();
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/move_test_code");
    let tx_data = test_cluster
        .test_transaction_builder()
        .await
        .publish(path)
        .build();
    let effects = test_cluster
        .sign_and_execute_transaction(&tx_data)
        .await
        .effects
        .unwrap();
    let mut coin = None;
    let mut deny_cap = None;
    let mut coin_package = None;
    for created in effects.created() {
        let object = test_cluster
            .get_object_from_fullnode_store(&created.object_id())
            .await
            .unwrap();
        if object.is_package() {
            coin_package = Some(object);
            continue;
        }
        if object.is_immutable() {
            continue;
        }
        if object.is_coin() {
            coin = Some(object);
        } else {
            deny_cap = Some(object);
        }
    }
    let new_coin = coin.unwrap();
    let deny_cap_object = deny_cap.unwrap();
    let coin_package = coin_package.unwrap();
    let deny_cap = deny_cap_object.to_rust::<FreezeCap>().unwrap();
    assert_eq!(deny_cap.package.bytes, coin_package.id());
    let sender = deny_cap_object.owner.get_address_owner_address().unwrap();
    let test_context = TestContext {
        coin_deny_list_object_init_version,
        coin_package: coin_package.id(),
        new_coin,
        deny_cap_object,
        sender,
    };
    (test_cluster, test_context)
}

async fn deny_address(test_cluster: &TestCluster, address: SuiAddress, test_context: &TestContext) {
    let gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(test_context.sender)
        .await
        .unwrap()
        .unwrap();
    let tx_data = test_cluster
        .test_transaction_builder_with_gas_object(test_context.sender, gas)
        .await
        .move_call(
            SUI_FRAMEWORK_PACKAGE_ID,
            COIN_DENY_LIST_MODULE.as_str(),
            "freeze_address",
            vec![
                CallArg::Object(ObjectArg::SharedObject {
                    id: COIN_DENY_LIST_OBJECT_ID,
                    initial_shared_version: test_context.coin_deny_list_object_init_version,
                    mutable: true,
                }),
                CallArg::Object(ObjectArg::ImmOrOwnedObject(
                    test_context.deny_cap_object.compute_object_reference(),
                )),
                CallArg::Pure(bcs::to_bytes(&address).unwrap()),
            ],
        )
        .with_type_args(vec![test_context.new_coin.coin_type_maybe().unwrap()])
        .build();
    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    info!("Deny effects: {:?}", response.effects.unwrap());
}
