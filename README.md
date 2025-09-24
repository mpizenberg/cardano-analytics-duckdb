# Cardano Fee Analytics

The goal of this project is to analyze the contribution of various Cardano projects to the network’s sustainability by generating transaction fees.
It is important to take decisions based on accurate data instead of assumptions and feelings.
This endeavor will help us understand the impact of different projects on the network's sustainability.

## Methodology

There are many approaches to tackle this problem.
Not all are equally effective.
We want to ensure that our analysis is comprehensive as well as easily reproducible by third parties.
The Cardano ecosystem is very fragmented, with many tools potentially viable and as many dead ends.
So an additional goal is to document different approaches to answer our questions, and report on the ease and sustainability of each.

Here are potential paths to attack this problem:
- Walk all blocks and process each Tx to filter its relevance and extract useful information.
- First index the chain to then enable fast access to relevant data via queries.

Chain processing, one Tx at a time is the most flexible approach.
We can use tools such as Ogmios, Oura or Yaci to process each transaction, keep or discard it, and store the relevant data in a database.
In contrast, using an indexer-only approach might be faster but limited to the data available in the index.
Since the goal is to provide analytics and not real-time data, full chain-processing might be the most adequate.

## Data - SNEK example

We need to analyze which transactions are likely due to a given project and the amount of fees they generate.
As you know, the UTxO model enables multi-asset transactions, but this also means that assets unrelated to a transaction intent are involved when spending the UTxO holding them.
Therefore, we cannot simply look for all transactions spending a given token.
Some projects also have build protocols (DeFi or otherwise) that do not involve specific tokens.
We will therefore try to map the different filters relevant to our analysis by examining in details the projects involved.
The observed patterns will then serve as a basis to generalize the approach in order to enable the analysis of other projects.

In the concrete case of the SNEK project, we can start by looking for the following types of transactions:

- Transactions in a DEX involving the SNEK token
- Transactions in a lending protocol involving the SNEK token
- Transactions related to the snek.fun protocol

### Transactions involving the SNEK token

Let’s start by gathering all transactions where the SNEK token moves from one address to another different.


## Struggles

- Dolos doesn’t work with recent protocol versions: https://github.com/txpipe/dolos/issues/486
- Dolos doesn’t seem to work with findIntersection calls in Ogmios: https://github.com/txpipe/dolos/issues/667
- Same problem with Kupo, getting IntersectionNotFound error.
- Dolos mini-blockfrost does not have the `/assets/{asset}/transactions` endpoint, making it hard to retrieve the list of transactions involving a given asset: https://docs.blockfrost.io/#tag/cardano--assets/GET/assets/{asset}/transactions
- Dolos step "importing immutable db" after mithril snapshot download is excruciatingly slow, taking more than 10h, with barely 2% CPU utilization.
