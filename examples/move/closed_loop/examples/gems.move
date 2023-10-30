// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

/// This is a simple example of a permissionless module for an imaginary game
/// that sells swords for Gems. Gems are an in-game currency that can be bought
/// with SUI.
module 0x0::sword {
    use sui::tx_context::TxContext;
    use sui::object::{Self, UID};

    use closed_loop::closed_loop::{Self as cl, Token, ActionRequest};
    use examples::gem::GEM;

    /// A game item that can be purchased with Gems.
    struct Sword has key, store { id: UID }

    /// Purchase a sword with Gems.
    public fun buy_sword(
        gems: Token<GEM>, ctx: &mut TxContext
    ): (Sword, ActionRequest<GEM>) {
        (
            Sword { id: object::new(ctx) },
            cl::spend(gems, ctx)
        )
    }
}

/// Module that defines the in-game currency: GEMs which can be purchased with
/// SUI and used to buy swords (in the `sword` module).
module examples::gem {
    use std::option::none;
    use std::string::{Self, String};
    use sui::sui::SUI;
    use sui::transfer;
    use sui::object::{Self, UID};
    use sui::balance::{Self, Balance};
    use sui::tx_context::{sender, TxContext};
    use sui::coin::{Self, Coin, TreasuryCap};

    use closed_loop::closed_loop::{Self as cl, Token, ActionRequest};

    /// Trying to purchase Gems with an unexpected amount.
    const EUnknownAmount: u64 = 0;

    /// 10 SUI is the price of a small bundle of Gems.
    const SMALL_BUNDLE: u64 = 10_000_000_000;
    const SMALL_AMOUNT: u64 = 100;

    /// 100 SUI is the price of a medium bundle of Gems.
    const MEDIUM_BUNDLE: u64 = 100_000_000_000;
    const MEDIUM_AMOUNT: u64 = 5_000;

    /// 1000 SUI is the price of a large bundle of Gems.
    /// This is the best deal.
    const LARGE_BUNDLE: u64 = 1_000_000_000_000;
    const LARGE_AMOUNT: u64 = 100_000;

    /// Gems can be purchased through the `Store`.
    struct GemStore has key {
        id: UID,
        /// Profits from selling Gems.
        profits: Balance<SUI>,
        /// The Treasury Cap for the in-game currency.
        gem_treasury: TreasuryCap<GEM>,
    }

    /// The OTW to create the in-game currency.
    struct GEM has drop {}

    // In the module initializer we create the in-game currency and define the
    // rules for different types of actions.
    fun init(otw: GEM, ctx: &mut TxContext) {
        let (treasury_cap, coin_metadata) = coin::create_currency(
            otw, 0, b"GEM", b"Capy Gems", // otw, decimal, symbol, name
            b"In-game currency for Capy Miners", // description
            none(), ctx // url
        );


        // create a `TokenPolicy` for GEMs
        let (policy, cap) = cl::new(&mut treasury_cap, ctx);

        cl::allow(&mut policy, &cap, buy_action(), ctx);
        cl::allow(&mut policy, &cap, cl::spend_action(), ctx);

        // create and share the GemStore
        transfer::share_object(GemStore {
            id: object::new(ctx),
            gem_treasury: treasury_cap,
            profits: balance::zero()
        });

        // deal with `TokenPolicy`, `CoinMetadata` and `TokenPolicyCap`
        transfer::public_freeze_object(coin_metadata);
        transfer::public_transfer(cap, sender(ctx));
        cl::share_policy(policy);
    }

    /// Purchase Gems from the GemStore. Very silly value matching against module
    /// constants...
    public fun buy_gems(
        self: &mut GemStore, payment: Coin<SUI>, ctx: &mut TxContext
    ): (Token<GEM>, ActionRequest<GEM>) {
        let amount = coin::value(&payment);
        let purchased = if (amount == SMALL_BUNDLE) {
            SMALL_AMOUNT
        } else if (amount == MEDIUM_BUNDLE) {
            MEDIUM_AMOUNT
        } else if (amount == LARGE_BUNDLE) {
            LARGE_AMOUNT
        } else {
            abort EUnknownAmount
        };

        coin::put(&mut self.profits, payment);

        // create custom request and mint some Gems
        let gems = cl::mint(&mut self.gem_treasury, purchased, ctx);
        let req = cl::new_request(buy_action(), purchased, none(), none(), ctx);

        (gems, req)
    }

    /// The name of the `buy` action in the `GemStore`.
    public fun buy_action(): String { string::utf8(b"buy") }
}