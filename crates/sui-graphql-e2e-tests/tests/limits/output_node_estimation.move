// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//# init --addresses A=0x42 --simulator

//# run-graphql --show-usage
# if connection does not have 'first' or 'last' set, use default_page_size (20)
{
  transactionBlockConnection {
    edges {
      txns: node {
        digest
      }
    }
  }
}

//# run-graphql --show-usage
# build on previous example with nested connection
{
  checkpointConnection {
    checkpoints: nodes {
      transactionBlockConnection {
        edges {
          txns: node {
            digest
          }
        }
      }
    }
  }
}

//# run-graphql --show-usage
# handles 1
{
  checkpointConnection {
    checkpoints: nodes {
      notOne: transactionBlockConnection {
        edges {
          txns: node {
            digest
          }
        }
      }
      isOne: transactionBlockConnection(first: 1) {
        edges {
          txns: node {
            digest
          }
        }
      }
    }
  }
}

//# run-graphql --show-usage
# handles 0
{
  checkpointConnection {
    checkpoints: nodes {
      notZero: transactionBlockConnection {
        edges {
          txns: node {
            digest
          }
        }
      }
      isZero: transactionBlockConnection(first: 0) {
        edges {
          txns: node {
            digest
          }
        }
      }
    }
  }
}

//# run-graphql --show-usage
# if connection does have 'first' set, use it
{
  transactionBlockConnection(first: 1) {
    edges {
      txns: node {
        digest
      }
    }
  }
}

//# run-graphql --show-usage
# if connection does have 'last' set, use it
{
  transactionBlockConnection(last: 1) {
    edges {
      txns: node {
        digest
      }
    }
  }
}

//# run-graphql --show-usage
# first and last should behave the same - total of 20 + 20*20 + 20*20 = 820
{
  transactionBlockConnection { # 20
    edges {
      txns: node {
        digest
        first: expiration {
          checkpointConnection(first: 20) { # 20 * 20
            edges {
              node {
                sequenceNumber
              }
            }
          }
        }
        last: expiration {
          checkpointConnection(last: 20) { # 20 * 20
            edges {
              node {
                sequenceNumber
              }
            }
          }
        }
      }
    }
  }
}

//# run-graphql --show-usage
# check that nodes have same behavior as edges
# first and last should behave the same - total of 20 + 20*20 + 20*20 = 820
{
  transactionBlockConnection {
    nodes {
      digest
      first: expiration {
        checkpointConnection(first: 20) {
          edges {
            node {
              sequenceNumber
            }
          }
        }
      }
      last: expiration {
        checkpointConnection(last: 20) {
          edges {
            node {
              sequenceNumber
            }
          }
        }
      }
    }
  }
}

//# run-graphql --show-usage
# example lifted from complex query at
# https://docs.github.com/en/graphql/overview/rate-limits-and-node-limits-for-the-graphql-api#node-limit
# 50 + (50 * 20) + (50 * 20 * 10) + (50 * 20) + (50 * 20 * 10) + 10 = 22060
{
  transactionBlockConnection(first: 50) { # 50
    edges {
      txns: node {
        digest
        a: expiration {
          checkpointConnection(last: 20) { # 50 * 20
            edges {
              checkpoints: node {
                transactionBlockConnection(first: 10) { # 50 * 20 * 10
                  edges {
                    checkpointTxns: node {
                      digest
                    }
                  }
                }
              }
            }
          }
        }
        b: expiration {
          checkpointConnection(first: 20) { # 50 * 20
            edges {
              checkpoints: node {
                transactionBlockConnection(last: 10) { # 50 * 20 * 10
                  edges {
                    checkpointTxns: node {
                      digest
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  eventConnection(last: 10) { # 10
    edges {
      event: node {
        timestamp
      }
    }
  }
}

//# run-graphql --show-usage
# error state - variable provided without accompanying value
query simpleOutputEstimation($howMany: Int) {
  transactionBlockConnection(last: $howMany) {
    edges {
      txns: node {
        digest
        a: expiration {
          checkpointConnection { # 50
            edges {
              checkpoints: node {
                transactionBlockConnection(first: $howMany) { # 20
                  edges {
                    checkpointTxns: node {
                      digest
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

//# run-graphql --show-usage
# error state - can't use first and last together
{
  transactionBlockConnection(first: 20, last: 30) {
    edges {
      txns: node {
        digest
      }
    }
  }
}
