# Cardano Analytics

The goal of this project is to make analytics on Cardano easier, more efficient, and more accessible to the community.
To that aim, we are defining a standard format based on Parquet files, and accessible with DuckDB.
Parquet files will be organized in folders, where each folder matches a range of 200000 slots (roughly 55 hours), for example `slot_15000000_15199999/`.
Inside each folder, we will have the following files:
- `tx-raw.parquet` containing raw transaction data
- `tx.parquet` containing basic information about transactions
- `utxo.parquet` containing basic information about UTxOs
- `asset.parquet` containing basic information about native assets in transactions
- `mint.parquet` containing basic information about minting in transactions
- `cert.parquet` containing basic information about certificates in transactions
- `vote.parquet` containing basic information about votes in transactions
- `proposal.parquet` containing basic information about proposals in transactions
- `redeemer.parquet` containing basic information about redeemers in transactions

## Parquet Schemas

Here are the detailed types needed for each of the aforementioned parquet files.

- `tx-raw.parquet`:
  - `raw_bytes`: BYTE_ARRAY

- `tx.parquet`:
  - `tx_id`: FIXED_LEN_BYTE_ARRAY(32)
  - `block_height`: INT32
  - `slot`: INT64
  - `fee`: INT64
  - `ref_input_count`: INT32
  - `input_count`: INT32
  - `output_count`: INT32
  - `redeemer_count`: INT32
  - `has_mint`: BOOLEAN
  - `has_withdrawal`: BOOLEAN
  - `has_cert`: BOOLEAN
  - `has_vote`: BOOLEAN
  - `has_proposal`: BOOLEAN

- `utxo.parquet`:
  - `tx_id`: FIXED_LEN_BYTE_ARRAY(32)
  - `output_index`: INT32
  - `address`: ASCII STRING
  - `lovelace`: INT64
  - `is_script_address`: BOOLEAN
  - `has_token`: BOOLEAN
  - `has_datum`: BOOLEAN
  - `has_ref_script`: BOOLEAN

- `asset.parquet`:
  - `tx_id`: FIXED_LEN_BYTE_ARRAY(32)
  - `output_index`: INT32
  - `policy_id`: FIXED_LEN_BYTE_ARRAY(28)
  - `asset_name`: ASCII STRING
  - `quantity`: INT64

- `mint.parquet`:
  - `tx_id`: FIXED_LEN_BYTE_ARRAY(32)
  - `policy_id`: FIXED_LEN_BYTE_ARRAY(28)
  - `asset_name`: ASCII STRING
  - `quantity`: INT64

- `cert.parquet`:
  - `tx_id`: FIXED_LEN_BYTE_ARRAY(32)
  - `cert_index`: INT32
  - `type`: ASCII STRING

- `vote.parquet`:
  - `tx_id`: FIXED_LEN_BYTE_ARRAY(32)
  - `voter`: ASCII STRING
  - `action_id_tx`: FIXED_LEN_BYTE_ARRAY(32)
  - `action_id_index`: INT32
  - `vote`: INT32
  - `anchor`: ASCII STRING

- `proposal.parquet`:
  - `tx_id`: FIXED_LEN_BYTE_ARRAY(32)
  - `proposal_index`: INT32
  - `deposit`: INT64
  - `return_account`: ASCII STRING
  - `type`: INT32
  - `anchor`: ASCII STRING

- `redeemer.parquet`:
  - `tx_id`: FIXED_LEN_BYTE_ARRAY(32)
  - `tag`: INT32
  - `redeemer_index`: INT32
  - `data`: BYTE_ARRAY
  - `mem`: INT64
  - `steps`: INT64

## Generation of the Parquet Files

Eventually, an efficient method to generate these files will be needed.
Currently, we are working on a small subset of the history, so we are using Ogmios to generate the data.
