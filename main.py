import ogmios
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from config import (
    ExtractionConfig,
    get_default_config,
    get_high_fee_config,
    PRESET_STARTING_POINTS,
)


def get_slot_group_directory(slot: int, group_size: int = 10000) -> str:
    """Get the directory name for a slot group."""
    group = slot // group_size
    return f"slot_{group * group_size}_{(group + 1) * group_size - 1}"


def extract_transaction_data(
    tx: Dict[str, Any], slot: int, config: ExtractionConfig
) -> Optional[Dict[str, Any]]:
    """Extract relevant data from a transaction, applying filters."""
    tx_id = tx.get("id", "")
    tx_fee = tx.get("fee", {}).get("ada", {}).get("lovelace", 0)
    tx_fee_ada = tx_fee / 1_000_000

    # Extract inputs (tx_id + output_index)
    inputs = []
    if tx.get("inputs"):
        for input_utxo in tx["inputs"]:
            input_tx_id = input_utxo.get("transaction", {}).get("id", "")
            output_index = input_utxo.get("index", 0)
            inputs.append(f"{input_tx_id}#{output_index}")

    # Extract output addresses
    output_addresses = []
    if tx.get("outputs"):
        for output in tx["outputs"]:
            address = output.get("address", "")
            if address:
                output_addresses.append(address)

    # Apply fee filter if specified
    if config.min_fee_ada and tx_fee_ada < config.min_fee_ada:
        return None

    # Apply address filter if specified
    if config.target_addresses:
        all_addresses = output_addresses.copy()
        # Add input addresses by checking transaction inputs (simplified)
        if not any(addr in all_addresses for addr in config.target_addresses):
            return None

    return {
        "slot": slot,
        "tx_id": tx_id,
        "tx_fee_lovelace": tx_fee,
        "tx_fee_ada": tx_fee_ada,
        "inputs": "|".join(inputs),  # Join with pipe separator for easy parsing
        "output_addresses": "|".join(output_addresses),  # Join with pipe separator
        "num_inputs": len(inputs),
        "num_outputs": len(output_addresses),
    }


def save_transactions_to_parquet(
    transactions: List[Dict[str, Any]], slot_group_dir: str, config: ExtractionConfig
):
    """Save transaction data to parquet file in the appropriate slot group directory."""
    if not transactions:
        return

    # Create the full directory path
    duckdb_dir = Path(config.output_dir)
    group_dir = duckdb_dir / slot_group_dir
    group_dir.mkdir(parents=True, exist_ok=True)

    # Create DataFrame and save to parquet
    df = pd.DataFrame(transactions)
    parquet_file = group_dir / "tx.parquet"

    # If file exists, append to it, otherwise create new
    if parquet_file.exists():
        existing_df = pd.read_parquet(parquet_file)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        # Remove duplicates based on tx_id
        combined_df = combined_df.drop_duplicates(subset=["tx_id"], keep="last")
        combined_df.to_parquet(parquet_file, index=False)
        print(
            f"Appended {len(df)} transactions to {parquet_file} (total: {len(combined_df)})"
        )
    else:
        df.to_parquet(parquet_file, index=False)
        print(f"Created {parquet_file} with {len(df)} transactions")


def main(config: Optional[ExtractionConfig] = None):
    """Main extraction function with configurable parameters."""
    if config is None:
        config = get_default_config()

    print("Starting Cardano fee analytics data extraction...")
    print(
        f"Configuration: batch_size={config.batch_size}, buffer_size={config.buffer_size}"
    )
    print(f"Output directory: {config.output_dir}")
    if config.min_fee_ada:
        print(f"Fee filter: Only transactions with fees >= {config.min_fee_ada} ADA")
    if config.target_addresses:
        print(f"Address filter: {len(config.target_addresses)} target addresses")

    transactions_buffer = {}  # slot_group -> list of transactions

    with ogmios.Client(host=config.ogmios_host, port=config.ogmios_port) as client:
        print("Connected to Ogmios client")

        # Set chain pointer based on configuration
        if config.start_point:
            point, tip, _ = client.find_intersection.execute([config.start_point])
            print(f"Starting from slot {point.slot}")
        else:
            print("No starting point specified, using chain tip")
            point, tip, _ = client.find_intersection.execute([ogmios.Origin()])

        total_txs_processed = 0
        blocks_processed = 0

        while True:
            # Batch requests to improve performance
            for i in range(config.batch_size):
                client.next_block.send()

            for i in range(config.batch_size):
                direction, tip, block, id = client.next_block.receive()
                if direction.value == "forward":
                    blocks_processed += 1
                    current_slot = getattr(block, "slot", 0)

                    if blocks_processed % config.progress_interval == 0:
                        print(
                            f"Processed {blocks_processed} blocks, current slot: {current_slot}, total txs: {total_txs_processed}"
                        )

                    # Process transactions in the block
                    if isinstance(block, ogmios.Block) and hasattr(
                        block, "transactions"
                    ):
                        slot_group_dir = get_slot_group_directory(
                            current_slot, config.slot_group_size
                        )

                        for tx in block.transactions:
                            tx_data = extract_transaction_data(tx, current_slot, config)
                            total_txs_processed += 1

                            # Skip if transaction doesn't match filters
                            if tx_data is None:
                                continue

                            # Add to buffer for this slot group
                            if slot_group_dir not in transactions_buffer:
                                transactions_buffer[slot_group_dir] = []

                            transactions_buffer[slot_group_dir].append(tx_data)

                            # Save to parquet if buffer is full
                            if (
                                len(transactions_buffer[slot_group_dir])
                                >= config.buffer_size
                            ):
                                save_transactions_to_parquet(
                                    transactions_buffer[slot_group_dir],
                                    slot_group_dir,
                                    config,
                                )
                                transactions_buffer[slot_group_dir] = []

                    # Stop when we've reached the network tip
                    if tip.height == block.height:
                        print(f"Reached chain tip at slot {tip.slot}")
                        # Save any remaining transactions in buffers
                        for slot_group_dir, txs in transactions_buffer.items():
                            if txs:
                                save_transactions_to_parquet(
                                    txs, slot_group_dir, config
                                )

                        print(
                            f"Data extraction complete. Total transactions processed: {total_txs_processed}"
                        )
                        print(
                            f"Data saved in {config.output_dir}/ directory organized by slot groups"
                        )
                        return

                elif direction.value == "backward":
                    print("Encountered rollback, continuing...")


def query_high_fee_transactions():
    """Query transactions with fees > 2 ADA using DuckDB."""
    import duckdb

    print("\nQuerying transactions with fees > 2 ADA...")

    # Connect to DuckDB
    conn = duckdb.connect()

    # Query all parquet files in the duckdb directory
    query = """
    SELECT
        slot,
        tx_id,
        tx_fee_ada,
        tx_fee_lovelace,
        num_inputs,
        num_outputs
    FROM read_parquet('duckdb/*/tx.parquet')
    WHERE tx_fee_ada > 2.0
    ORDER BY tx_fee_ada DESC
    LIMIT 100;
    """

    try:
        result = conn.execute(query).fetchdf()
        print(f"Found {len(result)} transactions with fees > 2 ADA")
        if len(result) > 0:
            print("\nTop high-fee transactions:")
            print(result.to_string(index=False))

        # Summary statistics
        summary_query = """
        SELECT
            COUNT(*) as total_high_fee_txs,
            AVG(tx_fee_ada) as avg_fee_ada,
            MAX(tx_fee_ada) as max_fee_ada,
            MIN(tx_fee_ada) as min_fee_ada
        FROM read_parquet('duckdb/*/tx.parquet')
        WHERE tx_fee_ada > 2.0;
        """

        summary = conn.execute(summary_query).fetchdf()
        print(f"\nSummary of high-fee transactions (> 2 ADA):")
        print(summary.to_string(index=False))

    except Exception as e:
        print(f"Error querying data: {e}")
        print("Make sure to run the data extraction first to create the parquet files.")

    conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cardano Fee Analytics Data Extraction"
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="extract",
        choices=["extract", "query"],
        help="Command to run: extract or query",
    )
    parser.add_argument(
        "--config",
        choices=["default", "high-fee", "full-history"],
        default="default",
        help="Configuration preset to use",
    )
    parser.add_argument(
        "--min-fee", type=float, help="Minimum fee in ADA for extraction"
    )
    parser.add_argument(
        "--start-point",
        choices=list(PRESET_STARTING_POINTS.keys()),
        help="Starting point for extraction",
    )
    parser.add_argument("--batch-size", type=int, help="Batch size for processing")
    parser.add_argument(
        "--buffer-size", type=int, help="Buffer size before writing to parquet"
    )

    args = parser.parse_args()

    if args.command == "query":
        query_high_fee_transactions()
    else:
        # Set up configuration based on arguments
        if args.config == "high-fee":
            config = get_high_fee_config(args.min_fee or 2.0)
        elif args.config == "full-history":
            from config import get_full_history_config

            config = get_full_history_config()
        else:
            config = get_default_config()

        # Override config with command line arguments
        if args.min_fee:
            config.min_fee_ada = args.min_fee
        if args.start_point:
            config.start_point = PRESET_STARTING_POINTS[args.start_point]
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.buffer_size:
            config.buffer_size = args.buffer_size

        main(config)
