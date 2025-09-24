import duckdb
import pandas as pd
from pathlib import Path


def query_high_fee_transactions(min_fee_ada: float = 2.0):
    """Query transactions with fees above a specified threshold using DuckDB."""
    print(f"\nQuerying transactions with fees > {min_fee_ada} ADA...")

    # Check if parquet files exist
    duckdb_dir = Path("duckdb")
    if not duckdb_dir.exists():
        print(
            "Error: duckdb directory not found. Please run the data extraction first."
        )
        return

    parquet_files = list(duckdb_dir.rglob("tx.parquet"))
    if not parquet_files:
        print("Error: No parquet files found. Please run the data extraction first.")
        return

    print(f"Found {len(parquet_files)} parquet files to analyze")

    # Connect to DuckDB
    conn = duckdb.connect()

    try:
        # Query high-fee transactions
        high_fee_query = f"""
        SELECT
            slot,
            tx_id,
            tx_fee_ada,
            tx_fee_lovelace,
            num_inputs,
            num_outputs,
            LENGTH(inputs) - LENGTH(REPLACE(inputs, '|', '')) + 1 as input_count_check,
            LENGTH(output_addresses) - LENGTH(REPLACE(output_addresses, '|', '')) + 1 as output_count_check
        FROM read_parquet('duckdb/*/tx.parquet')
        WHERE tx_fee_ada > {min_fee_ada}
        ORDER BY tx_fee_ada DESC
        LIMIT 100;
        """

        result = conn.execute(high_fee_query).fetchdf()
        print(f"\nFound {len(result)} transactions with fees > {min_fee_ada} ADA")

        if len(result) > 0:
            print(f"\nTop {min(10, len(result))} highest fee transactions:")
            print(result.head(10).to_string(index=False))

            # Save full results to CSV
            output_file = f"high_fee_transactions_{min_fee_ada}ada.csv"
            result.to_csv(output_file, index=False)
            print(f"\nFull results saved to {output_file}")

        # Summary statistics for all transactions
        summary_query = """
        SELECT
            COUNT(*) as total_transactions,
            AVG(tx_fee_ada) as avg_fee_ada,
            MAX(tx_fee_ada) as max_fee_ada,
            MIN(tx_fee_ada) as min_fee_ada,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tx_fee_ada) as median_fee_ada,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY tx_fee_ada) as p95_fee_ada,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY tx_fee_ada) as p99_fee_ada
        FROM read_parquet('duckdb/*/tx.parquet');
        """

        summary = conn.execute(summary_query).fetchdf()
        print(f"\nOverall transaction fee statistics:")
        print(summary.to_string(index=False))

        # High-fee transaction statistics
        high_fee_summary_query = f"""
        SELECT
            COUNT(*) as high_fee_count,
            AVG(tx_fee_ada) as avg_high_fee_ada,
            MAX(tx_fee_ada) as max_high_fee_ada,
            MIN(tx_fee_ada) as min_high_fee_ada,
            SUM(tx_fee_ada) as total_high_fees_ada
        FROM read_parquet('duckdb/*/tx.parquet')
        WHERE tx_fee_ada > {min_fee_ada};
        """

        high_fee_summary = conn.execute(high_fee_summary_query).fetchdf()
        print(f"\nHigh-fee transaction statistics (> {min_fee_ada} ADA):")
        print(high_fee_summary.to_string(index=False))

        # Fee distribution by ranges
        fee_distribution_query = """
        SELECT
            CASE
                WHEN tx_fee_ada < 0.5 THEN '< 0.5 ADA'
                WHEN tx_fee_ada < 1.0 THEN '0.5 - 1.0 ADA'
                WHEN tx_fee_ada < 2.0 THEN '1.0 - 2.0 ADA'
                WHEN tx_fee_ada < 5.0 THEN '2.0 - 5.0 ADA'
                WHEN tx_fee_ada < 10.0 THEN '5.0 - 10.0 ADA'
                ELSE '> 10.0 ADA'
            END as fee_range,
            COUNT(*) as transaction_count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
        FROM read_parquet('duckdb/*/tx.parquet')
        GROUP BY fee_range
        ORDER BY MIN(tx_fee_ada);
        """

        fee_dist = conn.execute(fee_distribution_query).fetchdf()
        print(f"\nTransaction fee distribution:")
        print(fee_dist.to_string(index=False))

    except Exception as e:
        print(f"Error querying data: {e}")
    finally:
        conn.close()


def query_transactions_by_slot_range(start_slot: int, end_slot: int):
    """Query transactions within a specific slot range."""
    print(f"\nQuerying transactions between slots {start_slot} and {end_slot}...")

    conn = duckdb.connect()

    try:
        query = f"""
        SELECT
            slot,
            COUNT(*) as transaction_count,
            AVG(tx_fee_ada) as avg_fee,
            MAX(tx_fee_ada) as max_fee,
            SUM(CASE WHEN tx_fee_ada > 2.0 THEN 1 ELSE 0 END) as high_fee_count
        FROM read_parquet('duckdb/*/tx.parquet')
        WHERE slot BETWEEN {start_slot} AND {end_slot}
        GROUP BY slot
        ORDER BY slot;
        """

        result = conn.execute(query).fetchdf()
        print(f"Found data for {len(result)} slots in the specified range")

        if len(result) > 0:
            print(result.to_string(index=False))

    except Exception as e:
        print(f"Error querying slot range: {e}")
    finally:
        conn.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Query Cardano transaction fee data")
    parser.add_argument(
        "--min-fee",
        type=float,
        default=2.0,
        help="Minimum fee in ADA to filter transactions (default: 2.0)",
    )
    parser.add_argument(
        "--slot-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Query transactions within a specific slot range",
    )

    args = parser.parse_args()

    if args.slot_range:
        query_transactions_by_slot_range(args.slot_range[0], args.slot_range[1])
    else:
        query_high_fee_transactions(args.min_fee)


if __name__ == "__main__":
    main()
