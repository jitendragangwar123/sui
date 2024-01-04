// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use super::{string_input::impl_string_input, sui_address::SuiAddress};
use crate::data::{BoxedQuery, Db, DbBackend};
use async_graphql::*;
use diesel::{
    expression::{is_aggregate::No, ValidGrouping},
    query_builder::QueryFragment,
    sql_types::{Binary, Text},
    AppearsOnTable, BoolExpressionMethods, Expression, ExpressionMethods, QueryDsl, QuerySource,
    TextExpressionMethods,
};
use move_core_types::parser::{parse_address, parse_module_id, parse_type_tag};
use std::str::FromStr;
use sui_types::TypeTag;

/// GraphQL scalar containing a filter on types.
#[derive(Clone, Debug)]
pub(crate) enum TypeFilter {
    /// Filter the type by the package or module it's from.
    ByModule(ModuleFilter),

    /// If the type tag has type parameters, treat it as an exact filter on that instantiation,
    /// otherwise treat it as either a filter on all generic instantiations of the type, or an exact
    /// match on the type with no type parameters. E.g.
    ///
    ///  0x2::coin::Coin
    ///
    /// would match both 0x2::coin::Coin and 0x2::coin::Coin<0x2::sui::SUI>.
    ByType(TypeTag),
}

/// GraphQL scalar containing a filter on modules.
#[derive(Clone, Debug)]
pub(crate) enum ModuleFilter {
    /// Filter the module by the package it's from.
    ByPackage(SuiAddress),

    /// Exact match on the module.
    ByModule(SuiAddress, String),
}

#[derive(thiserror::Error, Debug)]
pub(crate) enum Error {
    #[error("Invalid filter, expected: {0}")]
    InvalidFormat(&'static str),
}

impl TypeFilter {
    /// Modify `query` to apply this filter to `field`, returning the new query.
    pub(crate) fn apply<E, QS, ST, GB>(
        &self,
        query: BoxedQuery<ST, QS, Db, GB>,
        field: E,
    ) -> BoxedQuery<ST, QS, Db, GB>
    where
        BoxedQuery<ST, QS, Db, GB>: QueryDsl,
        E: ExpressionMethods + TextExpressionMethods,
        E: Expression<SqlType = Text> + QueryFragment<DbBackend>,
        E: AppearsOnTable<QS> + ValidGrouping<(), IsAggregate = No>,
        E: Clone + Send + 'static,
        QS: QuerySource,
    {
        match self {
            TypeFilter::ByModule(ModuleFilter::ByPackage(p)) => {
                query.filter(field.like(format!("{p}::%")))
            }

            TypeFilter::ByModule(ModuleFilter::ByModule(p, m)) => {
                query.filter(field.like(format!("{p}::{m}::%")))
            }

            TypeFilter::ByType(TypeTag::Struct(tag)) if tag.type_params.is_empty() => {
                let f1 = field.clone();
                let f2 = field;
                let exact = tag.to_canonical_string(/* with_prefix */ true);
                let prefix = format!("{}<%", tag.to_canonical_display(/* with_prefix */ true));
                query.filter(f1.eq(exact).or(f2.like(prefix)))
            }

            TypeFilter::ByType(tag) => {
                let exact = tag.to_canonical_string(/* with_prefix */ true);
                query.filter(field.eq(exact))
            }
        }
    }
}

impl ModuleFilter {
    /// Modify `query` to apply this filter, treating `package` as the column containing the package
    /// address and `module` as the module containing the module name.
    pub(crate) fn apply<P, M, QS, ST, GB>(
        &self,
        query: BoxedQuery<ST, QS, Db, GB>,
        package: P,
        module: M,
    ) -> BoxedQuery<ST, QS, Db, GB>
    where
        BoxedQuery<ST, QS, Db, GB>: QueryDsl,
        P: ExpressionMethods + Expression<SqlType = Binary> + QueryFragment<DbBackend>,
        M: ExpressionMethods + Expression<SqlType = Text> + QueryFragment<DbBackend>,
        P: AppearsOnTable<QS> + ValidGrouping<(), IsAggregate = No>,
        M: AppearsOnTable<QS> + ValidGrouping<(), IsAggregate = No>,
        P: Send + 'static,
        M: Send + 'static,
        QS: QuerySource,
    {
        match self {
            ModuleFilter::ByPackage(p) => query.filter(package.eq(p.into_vec())),
            ModuleFilter::ByModule(p, m) => query
                .filter(package.eq(p.into_vec()))
                .filter(module.eq(m.clone())),
        }
    }
}

impl_string_input!(TypeFilter);
impl_string_input!(ModuleFilter);

impl FromStr for TypeFilter {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Error> {
        if let Ok(tag) = parse_type_tag(s) {
            Ok(TypeFilter::ByType(tag))
        } else if let Ok(filter) = ModuleFilter::from_str(s) {
            Ok(TypeFilter::ByModule(filter))
        } else {
            Err(Error::InvalidFormat(
                "package[::module[::type[<type_params>]]] or primitive type",
            ))
        }
    }
}

impl FromStr for ModuleFilter {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Error> {
        if let Ok(module) = parse_module_id(s) {
            Ok(ModuleFilter::ByModule(
                SuiAddress::from(*module.address()),
                module.name().to_string(),
            ))
        } else if let Ok(package) = parse_address(s) {
            Ok(ModuleFilter::ByPackage(package.into()))
        } else {
            Err(Error::InvalidFormat("package[::module]"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use expect_test::expect;

    #[test]
    fn test_valid_filters() {
        let inputs = [
            "u8",
            "address",
            "bool",
            "0x2",
            "0x2::coin",
            "0x2::coin::Coin",
            "0x2::coin::Coin<0x2::sui::SUI>",
            "vector<u256>",
            "vector<0x3::staking_pool::StakedSui>",
        ]
        .into_iter();

        let filters: Vec<_> = inputs.map(|i| TypeFilter::from_str(i).unwrap()).collect();

        let expect = expect![[r#"
            [
                ByType(
                    U8,
                ),
                ByType(
                    Address,
                ),
                ByType(
                    Bool,
                ),
                ByModule(
                    ByPackage(
                        SuiAddress(
                            [
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                2,
                            ],
                        ),
                    ),
                ),
                ByModule(
                    ByModule(
                        SuiAddress(
                            [
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                2,
                            ],
                        ),
                        "coin",
                    ),
                ),
                ByType(
                    Struct(
                        StructTag {
                            address: 0000000000000000000000000000000000000000000000000000000000000002,
                            module: Identifier(
                                "coin",
                            ),
                            name: Identifier(
                                "Coin",
                            ),
                            type_params: [],
                        },
                    ),
                ),
                ByType(
                    Struct(
                        StructTag {
                            address: 0000000000000000000000000000000000000000000000000000000000000002,
                            module: Identifier(
                                "coin",
                            ),
                            name: Identifier(
                                "Coin",
                            ),
                            type_params: [
                                Struct(
                                    StructTag {
                                        address: 0000000000000000000000000000000000000000000000000000000000000002,
                                        module: Identifier(
                                            "sui",
                                        ),
                                        name: Identifier(
                                            "SUI",
                                        ),
                                        type_params: [],
                                    },
                                ),
                            ],
                        },
                    ),
                ),
                ByType(
                    Vector(
                        U256,
                    ),
                ),
                ByType(
                    Vector(
                        Struct(
                            StructTag {
                                address: 0000000000000000000000000000000000000000000000000000000000000003,
                                module: Identifier(
                                    "staking_pool",
                                ),
                                name: Identifier(
                                    "StakedSui",
                                ),
                                type_params: [],
                            },
                        ),
                    ),
                ),
            ]"#]];
        expect.assert_eq(&format!("{filters:#?}"))
    }

    #[test]
    fn test_invalid_type_filters() {
        for invalid_type_filters in [
            "not_a_real_type",
            "0x1:missing::colon",
            "0x2::trailing::",
            "0x3::mismatched::bra<0x4::ke::ts",
        ] {
            assert!(TypeFilter::from_str(invalid_type_filters).is_err());
        }
    }

    #[test]
    fn test_invalid_module_filters() {
        for invalid_module_filters in [
            "u8",
            "address",
            "bool",
            "0x2::coin::Coin",
            "0x2::coin::Coin<0x2::sui::SUI>",
            "vector<u256>",
            "vector<0x3::staking_pool::StakedSui>",
        ] {
            assert!(ModuleFilter::from_str(invalid_module_filters).is_err());
        }
    }
}
