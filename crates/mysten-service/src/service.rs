// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use crate::health::HealthResponse;
use crate::DEFAULT_PORT;
use anyhow::Result;
use axum::routing::get;
use axum::Json;
use axum::Router;

pub fn get_mysten_service(app_name: &str, app_version: &str) -> Router {
    // build our application with a single route
    Router::new().route(
        "/health",
        get(Json(HealthResponse::new(app_name, app_version))),
    )
}

pub async fn serve(app: Router) -> Result<()> {
    // run it with hyper on localhost:3000
    println!("listening on http://localhost:{}", DEFAULT_PORT);
    axum::Server::bind(&format!("0.0.0.0:{}", DEFAULT_PORT).parse()?)
        .serve(app.into_make_service())
        .await?;
    Ok(())
}
