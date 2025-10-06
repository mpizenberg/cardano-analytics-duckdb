# Cardano Analytics

The goal of this project is to make analytics on Cardano easier, more efficient, and more accessible to the community.
To that aim, we are defining a standard format based on Parquet files, and accessible with DuckDB.
Parquet files will be organized in folders, where each folder matches a range of 200000 slots (roughly 55 hours), for example `slot_15000000_15199999/`.
Inside each folder, we will have the following files:
- `tx-raw.parquet` containing raw transaction data
- `tx.parquet` containing basic information about transactions
- `utxo.parquet` containing basic information about UTxOs
- `mint.parquet` containing basic information about minting in transactions
- `cert.parquet` containing basic information about certificates in transactions
- `vote.parquet` containing basic information about votes in transactions
- `redeemer.parquet` containing basic information about redeemers in transactions
- `proposal.parquet` containing basic information about proposals in transactions

## Generation of the Parquet Files

Eventually, an efficient method to generate these files will be needed.
Currently, we are working on a small subset of the history, so we are using Ogmios to generate the data.
