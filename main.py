from tqdm import tqdm
from config import (
    ExtractionConfig,
    PRESET_STARTING_POINTS,
)
from ogmios_parquet import extract_transactions


def main(config: ExtractionConfig):
    """Main entry point that delegates to the extraction module."""
    extract_transactions(config)


def query_high_fee_transactions():
    """Query transactions with fees > 2 ADA using DuckDB."""
    import duckdb

    tqdm.write("\nQuerying transactions with fees > 2 ADA...")

    # Connect to DuckDB
    conn = duckdb.connect()

    # Query all parquet files in the duckdb directory
    query = """
    SELECT
        slot,
        tx_id,
        tx_fee,
        num_inputs,
        num_outputs
    FROM read_parquet('duckdb/*/tx.parquet')
    WHERE tx_fee > 2000000
    ORDER BY tx_fee DESC
    LIMIT 100;
    """

    try:
        result = conn.execute(query).fetchdf()
        tqdm.write(f"Found {len(result)} transactions with fees > 2 ADA")
        if len(result) > 0:
            tqdm.write("\nTop high-fee transactions:")
            tqdm.write(result.to_string(index=False))

        # Summary statistics
        summary_query = """
        SELECT
            COUNT(*) as total_high_fee_txs,
            AVG(tx_fee) as avg_fee,
            MAX(tx_fee) as max_fee,
            MIN(tx_fee) as min_fee
        FROM read_parquet('duckdb/*/tx.parquet')
        WHERE tx_fee > 2000000;
        """

        summary = conn.execute(summary_query).fetchdf()
        tqdm.write("Summary of high-fee transactions (> 2 ADA):")
        tqdm.write(summary.to_string(index=False))

    except Exception as e:
        tqdm.write(f"Error querying data: {e}")
        tqdm.write(
            "Make sure to run the data extraction first to create the parquet files."
        )

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
        choices=["default", "performance", "full-history"],
        default="default",
        help="Configuration preset to use",
    )
    parser.add_argument(
        "--start-point",
        choices=list(PRESET_STARTING_POINTS.keys()),
        help="Starting point for extraction",
    )
    parser.add_argument(
        "--stop-point",
        choices=list(PRESET_STARTING_POINTS.keys()),
        help="Stopping point for extraction",
    )
    parser.add_argument("--batch-size", type=int, help="Batch size for processing")
    parser.add_argument(
        "--buffer-size-slots",
        type=int,
        help="Buffer size in slots before writing to parquet",
    )
    parser.add_argument(
        "--slot-group-size", type=int, help="Number of slots per directory group"
    )

    args = parser.parse_args()

    if args.command == "query":
        query_high_fee_transactions()
    else:
        config = ExtractionConfig(
            # start_point=PRESET_STARTING_POINTS["last_alonzo"],
            # start_point=PRESET_STARTING_POINTS["snek_mint"],
            start_point=PRESET_STARTING_POINTS["last_byron"],
            # stop_point=PRESET_STARTING_POINTS["snek_mint"],snek_mint_plus_100K_blocks
            # stop_point=PRESET_STARTING_POINTS["snek_mint_plus_100K_blocks"],
            # stop_point=PRESET_STARTING_POINTS["last_babbage_plus_200K_blocks"],
            stop_point=PRESET_STARTING_POINTS["block_2025_10_28"],
        )
        # Override config with command line arguments
        if args.start_point:
            config.start_point = PRESET_STARTING_POINTS[args.start_point]
        if args.stop_point:
            config.stop_point = PRESET_STARTING_POINTS[args.stop_point]
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.buffer_size_slots:
            config.buffer_size_slots = args.buffer_size_slots
        if args.slot_group_size:
            config.slot_group_size = args.slot_group_size

        main(config)
