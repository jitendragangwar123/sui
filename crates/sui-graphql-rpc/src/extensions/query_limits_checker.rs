// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use crate::config::Limits;
use crate::config::ServiceConfig;
use crate::error::code;
use crate::error::code::INTERNAL_SERVER_ERROR;
use crate::error::graphql_error;
use crate::error::graphql_error_at_pos;
use crate::metrics::RequestMetrics;
use async_graphql::extensions::NextParseQuery;
use async_graphql::extensions::NextRequest;
use async_graphql::parser::types::ExecutableDocument;
use async_graphql::parser::types::Field;
use async_graphql::parser::types::FragmentDefinition;
use async_graphql::parser::types::Selection;
use async_graphql::parser::types::SelectionSet;
use async_graphql::value;
use async_graphql::Name;
use async_graphql::Pos;
use async_graphql::Positioned;
use async_graphql::Response;
use async_graphql::ServerResult;
use async_graphql::Value;
use async_graphql::Variables;
use async_graphql::{
    extensions::{Extension, ExtensionContext, ExtensionFactory},
    ServerError,
};
use async_graphql_value::Value as GqlValue;
use axum::headers;
use axum::http::HeaderName;
use axum::http::HeaderValue;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::Mutex;

pub static LIMITS_HEADER: HeaderName = HeaderName::from_static("x-sui-rpc-show-usage");

/// Only display usage information if this header was in the request.
pub(crate) struct ShowUsage;

#[derive(Clone, Debug, Default)]
struct ValidationRes {
    num_nodes: u32,
    output_nodes: u64,
    depth: u32,
    num_variables: u32,
    num_fragments: u32,
    query_payload: u32,
}

#[derive(Debug, Default)]
pub(crate) struct QueryLimitsChecker {
    validation_result: Mutex<Option<ValidationRes>>,
}

impl headers::Header for ShowUsage {
    fn name() -> &'static HeaderName {
        &LIMITS_HEADER
    }

    fn decode<'i, I>(_: &mut I) -> Result<Self, headers::Error>
    where
        I: Iterator<Item = &'i HeaderValue>,
    {
        Ok(ShowUsage)
    }

    fn encode<E: Extend<HeaderValue>>(&self, _: &mut E) {
        unimplemented!()
    }
}

impl ExtensionFactory for QueryLimitsChecker {
    fn create(&self) -> Arc<dyn Extension> {
        Arc::new(QueryLimitsChecker {
            validation_result: Mutex::new(None),
        })
    }
}

#[derive(Debug)]
struct ComponentCost {
    pub num_nodes: u32,
    pub depth: u32,
    pub output_nodes: u64,
}

impl std::ops::Add for ComponentCost {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            num_nodes: self.num_nodes + rhs.num_nodes,
            depth: self.depth + rhs.depth,
            output_nodes: self.output_nodes + rhs.output_nodes,
        }
    }
}

#[async_trait::async_trait]
impl Extension for QueryLimitsChecker {
    async fn request(&self, ctx: &ExtensionContext<'_>, next: NextRequest<'_>) -> Response {
        let resp = next.run(ctx).await;
        let validation_result = self.validation_result.lock().await.take();
        if let Some(validation_result) = validation_result {
            resp.extension(
                "usage",
                value! ({
                    "nodes": validation_result.num_nodes,
                    "output_nodes": validation_result.output_nodes,
                    "depth": validation_result.depth,
                    "variables": validation_result.num_variables,
                    "fragments": validation_result.num_fragments,
                    "query_payload": validation_result.query_payload,
                }),
            )
        } else {
            resp
        }
    }

    /// Validates the query against the limits set in the service config
    /// If the limits are hit, the operation terminates early
    async fn parse_query(
        &self,
        ctx: &ExtensionContext<'_>,
        query: &str,
        variables: &Variables,
        next: NextParseQuery<'_>,
    ) -> ServerResult<ExecutableDocument> {
        let cfg = ctx
            .data::<ServiceConfig>()
            .expect("No service config provided in schema data");

        if query.len() > cfg.limits.max_query_payload_size as usize {
            return Err(graphql_error(
                code::GRAPHQL_VALIDATION_FAILED,
                format!(
                    "Query payload is too large. The maximum allowed is {} bytes",
                    cfg.limits.max_query_payload_size
                ),
            ));
        }

        // Document layout of the query
        let doc = next.run(ctx, query, variables).await?;

        // TODO: Limit the complexity of fragments early on

        let mut running_costs = ComponentCost {
            depth: 0,
            num_nodes: 0,
            output_nodes: 0,
        };
        let mut max_depth_seen = 0;

        // An operation is a query, mutation or subscription consisting of a set of selections
        for (count, (_name, oper)) in doc.operations.iter().enumerate() {
            let sel_set = &oper.node.selection_set;

            // If the query is pure introspection, we don't need to check the limits.
            // Pure introspection queries are queries that only have one operation with one field
            // and that field is a `__schema` query
            if (count == 0) && (sel_set.node.items.len() == 1) {
                if let Some(node) = sel_set.node.items.first() {
                    if let Selection::Field(field) = &node.node {
                        if field.node.name.node == "__schema" {
                            continue;
                        }
                    }
                }
            }

            running_costs.depth = 0;
            self.analyze_selection_set(
                &cfg.limits,
                &doc.fragments,
                sel_set,
                &mut running_costs,
                variables,
            )?;
            max_depth_seen = max_depth_seen.max(running_costs.depth);
        }

        if ctx.data_opt::<ShowUsage>().is_some() {
            *self.validation_result.lock().await = Some(ValidationRes {
                num_nodes: running_costs.num_nodes,
                output_nodes: running_costs.output_nodes,
                depth: running_costs.depth,
                query_payload: query.len() as u32,
                num_variables: variables.len() as u32,
                num_fragments: doc.fragments.len() as u32,
            });
        }
        if let Some(metrics) = ctx.data_opt::<Arc<RequestMetrics>>() {
            metrics.num_nodes.observe(running_costs.num_nodes as f64);
            metrics.query_depth.observe(running_costs.depth as f64);
            metrics.query_payload_size.observe(query.len() as f64);
        }
        Ok(doc)
    }
}

impl QueryLimitsChecker {
    /// Parse the selected fields in one operation and check if it conforms to configured limits.
    fn analyze_selection_set(
        &self,
        limits: &Limits,
        fragment_defs: &HashMap<Name, Positioned<FragmentDefinition>>,
        sel_set: &Positioned<SelectionSet>,
        cost: &mut ComponentCost,
        variables: &Variables,
    ) -> ServerResult<()> {
        // Use BFS to analyze the query and count the number of nodes and the depth of the query

        // Queue to store the nodes at each level
        let mut que = VecDeque::new();

        for top_level_sel in sel_set.node.items.iter() {
            que.push_back((
                top_level_sel,
                /* parent_node_count */ 1 as u64,
                /* parent_is_connection */ false,
            ));
            cost.num_nodes += 1;
            check_limits(
                limits,
                cost.num_nodes,
                cost.depth,
                cost.output_nodes,
                Some(top_level_sel.pos),
            )?;
        }

        // Track the number of nodes at first level if any
        let mut level_len = que.len();

        while !que.is_empty() {
            // Signifies the start of a new level
            cost.depth += 1;
            check_limits(limits, cost.num_nodes, cost.depth, cost.output_nodes, None)?;
            while level_len > 0 {
                // Ok to unwrap since we checked for empty queue
                // and level_len > 0
                let (curr_sel, parent_count, parent_is_connection) = que.pop_front().unwrap();

                match &curr_sel.node {
                    Selection::Field(f) => {
                        if !f.node.directives.is_empty() {
                            return Err(graphql_error_at_pos(
                                INTERNAL_SERVER_ERROR,
                                "Fields with directives are not supported",
                                f.pos,
                            ));
                        }

                        let (current_count, is_connection) = estimate_output_nodes_for_curr_node(
                            f,
                            variables,
                            parent_is_connection,
                            limits,
                        );

                        let final_count = parent_count * current_count;

                        // Only update the "global" tally if this is a connection
                        if is_connection {
                            cost.output_nodes += final_count;
                        }

                        for field_sel in f.node.selection_set.node.items.iter() {
                            que.push_back((field_sel, final_count, is_connection));
                            cost.num_nodes += 1;
                            check_limits(
                                limits,
                                cost.num_nodes,
                                cost.depth,
                                cost.output_nodes,
                                Some(field_sel.pos),
                            )?;
                        }
                    }

                    Selection::FragmentSpread(fs) => {
                        let frag_name = &fs.node.fragment_name.node;
                        let frag_def = fragment_defs.get(frag_name).ok_or_else(|| {
                            graphql_error_at_pos(
                                INTERNAL_SERVER_ERROR,
                                format!(
                                    "Fragment {} not found but present in fragment list",
                                    frag_name
                                ),
                                fs.pos,
                            )
                        })?;

                        // TODO: this is inefficient as we might loop over same fragment multiple times
                        // Ideally web should cache the costs of fragments we've seen before
                        // Will do as enhancement
                        if !frag_def.node.directives.is_empty() {
                            return Err(graphql_error_at_pos(
                                INTERNAL_SERVER_ERROR,
                                "Fragments with directives are not supported",
                                frag_def.pos,
                            ));
                        }
                        for frag_sel in frag_def.node.selection_set.node.items.iter() {
                            que.push_back((frag_sel, parent_count, parent_is_connection));
                            cost.num_nodes += 1;
                            check_limits(
                                limits,
                                cost.num_nodes,
                                cost.depth,
                                cost.output_nodes,
                                Some(frag_sel.pos),
                            )?;
                        }
                    }

                    Selection::InlineFragment(fs) => {
                        if !fs.node.directives.is_empty() {
                            return Err(graphql_error_at_pos(
                                INTERNAL_SERVER_ERROR,
                                "Inline fragments with directives are not supported",
                                fs.pos,
                            ));
                        }
                        for in_frag_sel in fs.node.selection_set.node.items.iter() {
                            que.push_back((in_frag_sel, parent_count, parent_is_connection));
                            cost.num_nodes += 1;
                            check_limits(
                                limits,
                                cost.num_nodes,
                                cost.depth,
                                cost.output_nodes,
                                Some(in_frag_sel.pos),
                            )?;
                        }
                    }
                }
                level_len -= 1;
            }
            level_len = que.len();
        }

        if cost.output_nodes < 1 {
            cost.output_nodes = 1;
        }
        Ok(())
    }
}

fn check_limits(
    limits: &Limits,
    nodes: u32,
    depth: u32,
    output_nodes: u64,
    pos: Option<Pos>,
) -> ServerResult<()> {
    if nodes > limits.max_query_nodes {
        return Err(ServerError::new(
            format!(
                "Query has too many nodes. The maximum allowed is {}",
                limits.max_query_nodes
            ),
            pos,
        ));
    }

    if depth > limits.max_query_depth {
        return Err(ServerError::new(
            format!(
                "Query has too many levels of nesting. The maximum allowed is {}",
                limits.max_query_depth
            ),
            pos,
        ));
    }

    if output_nodes > limits.max_output_nodes {
        return Err(ServerError::new(
            format!(
                "Query will result in too many output nodes. The maximum allowed is {}, estimated {}",
                limits.max_output_nodes, output_nodes
            ),
            pos,
        ));
    }

    Ok(())
}

/// Given a node, estimate the number of output nodes it will produce.
fn estimate_output_nodes_for_curr_node(
    f: &Positioned<Field>,
    variables: &Variables,
    parent_is_connection: bool,
    limits: &Limits,
) -> (u64, bool) {
    let mut current_count = 1;
    let mut is_connection = false;

    // If the parent connection has 'first' or 'last' set, parent_is_connection is true.
    // In that scenario, 'edges' and 'nodes' are normal, non-connection fields.
    // Otherwise, this node is considered a connection and has count set to default_page_size.
    if f.node.name.node == "edges" || f.node.name.node == "nodes" {
        if !parent_is_connection {
            current_count = limits.default_page_size;
            is_connection = true;
        }
    } else {
        // Check if the current node is a connection by checking if it has 'first' or 'last' selected as a field
        let first_arg = f.node.get_argument("first");
        let last_arg = f.node.get_argument("last");

        let first_count = extract_limit(first_arg, variables).unwrap_or(limits.default_page_size);
        let last_count = extract_limit(last_arg, variables).unwrap_or(limits.default_page_size);

        match (first_arg, last_arg) {
            (None, None) => { /* Not a connection, no further action required */ }
            (Some(_), None) => {
                current_count = first_count;
            }
            (None, Some(_)) => {
                current_count = last_count;
            }
            (Some(_), Some(_)) => {
                // This follows the cursor spec for when both args are provided
                // the resulting slice is the min of the two
                current_count = std::cmp::min(first_count, last_count);
            }
        }
        is_connection = true;
    }
    (current_count, is_connection)
}

/// Try to extract a u64 value from the given argument, or return None on failure.
fn extract_limit(value: Option<&Positioned<GqlValue>>, variables: &Variables) -> Option<u64> {
    let Some(value) = value else {
        return None;
    };

    if let GqlValue::Variable(var) = &value.node {
        return match variables.get(var) {
            Some(Value::Number(num)) => num.as_u64(),
            _ => None,
        };
    }

    let GqlValue::Number(value) = &value.node else {
        return None;
    };
    value.as_u64()
}
