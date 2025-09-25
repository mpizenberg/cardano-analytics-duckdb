#!/usr/bin/env python3
"""
Examples showing how to use the Cardano fee analytics system.

This script demonstrates different ways to extract and analyze transaction data
from the Cardano blockchain using the fee analytics tools.
"""

import sys
from pathlib import Path
from config import (
    ExtractionConfig,
    get_default_config,
    get_performance_config,
    PRESET_STARTING_POINTS,
    KNOWN_ADDRESSES,
)


def example_basic_extraction():
    """Example: Basic transaction data extraction."""
    print("=" * 50)
    print("Example 1: Basic Transaction Data Extraction")
    print("=" * 50)

    print("""
This example shows how to extract all transaction data starting from
the SNEK mint point using default settings.

Command:
    python main.py extract

This will:
- Start from the SNEK mint block (slot 90914081)
- Process blocks in batches of 10
- Save data to 'duckdb/' directory
- Organize files by slot groups of 10,000 slots
- Save transactions to parquet files every 1,000 slots
    """)


def example_performance_extraction():
    """Example: Performance optimized extraction."""
    print("=" * 50)
    print("Example 2: Performance Optimized Extraction")
    print("=" * 50)

    print("""
This example shows how to optimize extraction for better performance.

Command:
    python main.py extract --config performance

This will:
- Use larger batch sizes for better throughput
- Optimize buffer sizes for performance
- Use settings tuned for faster processing
- Ideal for processing large amounts of data
    """)


def example_custom_extraction():
    """Example: Custom extraction with specific parameters."""
    print("=" * 50)
    print("Example 3: Custom Extraction Parameters")
    print("=" * 50)

    print("""
This example shows how to customize the extraction process.

Command:
    python main.py extract --start-point last_shelley --batch-size 25 --buffer-size-slots 2000

Available starting points:
- last_byron: Last block before Shelley era
- last_shelley: Last block before Allegra era
- last_allegra: Last block before Mary era
- last_mary: Last block before Alonzo era
- last_alonzo: Last block before Babbage era
- last_babbage: Last block before Conway era
- snek_mint: Block before SNEK token mint

This allows you to:
- Start from any major era in Cardano's history
- Adjust batch size for your hardware/network capacity
- Control temporal buffer sizes in slots
    """)


def example_querying_data():
    """Example: Querying extracted data."""
    print("=" * 50)
    print("Example 4: Querying Extracted Data")
    print("=" * 50)

    print("""
Once you have extracted data, you can query it in several ways:

1. Using the built-in query command:
    python main.py query

2. Using the dedicated query script:
    python query_high_fees.py
    python query_high_fees.py --min-fee 5.0
    python query_high_fees.py --slot-range 90914081 90920000

3. Using DuckDB directly in Python:
    """)

    print('''
import duckdb

conn = duckdb.connect()

# Find transactions with fees > 2 ADA
high_fee_txs = conn.execute("""
    SELECT tx_id, tx_fee_ada, slot, num_inputs, num_outputs
    FROM read_parquet('duckdb/*/tx.parquet')
    WHERE tx_fee_ada > 2.0
    ORDER BY tx_fee_ada DESC
    LIMIT 10
""").fetchdf()

print(high_fee_txs)

# Calculate average fees by time period
avg_fees_by_slot_group = conn.execute("""
    SELECT
        (slot // 10000) * 10000 as slot_group_start,
        COUNT(*) as tx_count,
        AVG(tx_fee_ada) as avg_fee_ada,
        MAX(tx_fee_ada) as max_fee_ada
    FROM read_parquet('duckdb/*/tx.parquet')
    GROUP BY slot_group_start
    ORDER BY slot_group_start
""").fetchdf()

print(avg_fees_by_slot_group)

conn.close()
    ''')


def example_custom_slot_groups():
    """Example: Custom slot group configurations."""
    print("=" * 50)
    print("Example 5: Custom Slot Group Configuration")
    print("=" * 50)

    print(f"""
This example shows how to configure slot groups and buffer sizes.

Known addresses for reference:
""")

    for name, addr in KNOWN_ADDRESSES.items():
        print(f"- {name}: {addr}")

    print("""
To create custom slot grouping:
    """)

    print("""
# Large slot groups for historical analysis
python main.py extract --slot-group-size 50000 --buffer-size-slots 5000

# Small slot groups for detailed analysis
python main.py extract --slot-group-size 1000 --buffer-size-slots 100

# Custom configuration programmatically:
from config import ExtractionConfig
from main import main

config = ExtractionConfig(
    slot_group_size=25000,  # 25k slots per directory
    buffer_size_slots=2500, # Save every 2.5k slots
    batch_size=15
)

main(config)
    """)


def example_programmatic_usage():
    """Example: Using the system programmatically."""
    print("=" * 50)
    print("Example 6: Programmatic Usage")
    print("=" * 50)

    print("""
You can use the extraction system programmatically in your own scripts:
    """)

    print('''
from config import ExtractionConfig
from main import main
import ogmios

# Create a custom configuration
config = ExtractionConfig(
    start_point=ogmios.Point(
        slot=90914081,
        id="2f7784ab8eee0e3d81223b9bd482195617cbee662ed6c412b123568251aac67a"
    ),
    batch_size=20,
    buffer_size_slots=500,
    output_dir="my_custom_data",
    progress_interval=50
)

# Run extraction
main(config)

# Then analyze the data
import duckdb
conn = duckdb.connect()

result = conn.execute("""
    SELECT
        COUNT(*) as total_transactions,
        AVG(tx_fee_ada) as avg_fee,
        SUM(tx_fee_ada) as total_fees_ada
    FROM read_parquet('my_custom_data/*/tx.parquet')
""").fetchone()

print(f"Total transactions: {result[0]}")
print(f"Average fee: {result[1]:.6f} ADA")
print(f"Total fees collected: {result[2]:.2f} ADA")
    ''')


def example_analyzing_fee_trends():
    """Example: Analyzing fee trends over time."""
    print("=" * 50)
    print("Example 7: Fee Trend Analysis")
    print("=" * 50)

    print("""
Analyze how transaction fees change over time:
    """)

    print('''
import duckdb
import pandas as pd
import matplotlib.pyplot as plt

conn = duckdb.connect()

# Get fee trends by day (assuming ~86400 seconds per day, ~20 seconds per slot)
daily_trends = conn.execute("""
    SELECT
        (slot // 4320) * 4320 as day_start_slot,  -- ~4320 slots per day
        COUNT(*) as daily_tx_count,
        AVG(tx_fee_ada) as avg_daily_fee,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tx_fee_ada) as median_daily_fee,
        MAX(tx_fee_ada) as max_daily_fee,
        COUNT(CASE WHEN tx_fee_ada > 2.0 THEN 1 END) as high_fee_count
    FROM read_parquet('duckdb/*/tx.parquet')
    GROUP BY day_start_slot
    ORDER BY day_start_slot
""").fetchdf()

# Plot fee trends
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(daily_trends['day_start_slot'], daily_trends['avg_daily_fee'])
plt.title('Average Daily Transaction Fees')
plt.ylabel('Average Fee (ADA)')

plt.subplot(2, 2, 2)
plt.plot(daily_trends['day_start_slot'], daily_trends['daily_tx_count'])
plt.title('Daily Transaction Count')
plt.ylabel('Transactions per Day')

plt.subplot(2, 2, 3)
plt.plot(daily_trends['day_start_slot'], daily_trends['high_fee_count'])
plt.title('High Fee Transactions (>2 ADA) per Day')
plt.ylabel('High Fee Transaction Count')

plt.subplot(2, 2, 4)
plt.plot(daily_trends['day_start_slot'], daily_trends['max_daily_fee'])
plt.title('Maximum Daily Fee')
plt.ylabel('Max Fee (ADA)')
plt.yscale('log')

plt.tight_layout()
plt.savefig('fee_trends.png')
plt.show()

conn.close()
    ''')


def example_complex_queries():
    """Example: Complex analytical queries."""
    print("=" * 50)
    print("Example 8: Complex Analytical Queries")
    print("=" * 50)

    print("""
Perform complex analysis using DuckDB's advanced features:
    """)

    print('''
import duckdb

conn = duckdb.connect()

# Find transactions with unusual input/output ratios and high fees
unusual_txs = conn.execute("""
    SELECT
        tx_id,
        tx_fee_ada,
        num_inputs,
        num_outputs,
        tx_fee_ada / num_inputs as fee_per_input,
        slot
    FROM read_parquet('duckdb/*/tx.parquet')
    WHERE tx_fee_ada > 1.0
      AND num_inputs > 0
      AND (num_outputs / CAST(num_inputs AS FLOAT)) > 5  -- Many outputs per input
    ORDER BY fee_per_input DESC
    LIMIT 20
""").fetchdf()

# Analyze fee distribution patterns
fee_patterns = conn.execute("""
    WITH fee_stats AS (
        SELECT
            tx_fee_ada,
            num_inputs,
            num_outputs,
            LENGTH(inputs) - LENGTH(REPLACE(inputs, '|', '')) + 1 as actual_input_count,
            CASE
                WHEN num_inputs <= 2 THEN 'simple'
                WHEN num_inputs <= 10 THEN 'moderate'
                ELSE 'complex'
            END as tx_complexity
        FROM read_parquet('duckdb/*/tx.parquet')
        WHERE tx_fee_ada > 0
    )
    SELECT
        tx_complexity,
        COUNT(*) as tx_count,
        AVG(tx_fee_ada) as avg_fee,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tx_fee_ada) as median_fee,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY tx_fee_ada) as p95_fee,
        MAX(tx_fee_ada) as max_fee
    FROM fee_stats
    GROUP BY tx_complexity
    ORDER BY avg_fee
""").fetchdf()

print("Fee patterns by transaction complexity:")
print(fee_patterns)

# Find potential batching or multi-sig patterns
batching_analysis = conn.execute("""
    SELECT
        num_outputs,
        COUNT(*) as tx_count,
        AVG(tx_fee_ada) as avg_fee,
        AVG(tx_fee_ada / num_outputs) as avg_fee_per_output
    FROM read_parquet('duckdb/*/tx.parquet')
    WHERE num_outputs BETWEEN 5 AND 100
      AND tx_fee_ada > 0.5
    GROUP BY num_outputs
    HAVING COUNT(*) > 10
    ORDER BY num_outputs
""").fetchdf()

print("\\nBatching patterns (transactions with 5-100 outputs):")
print(batching_analysis)

conn.close()
    ''')


def main():
    """Display all examples."""
    print("Cardano Fee Analytics - Usage Examples")
    print("=" * 60)
    print()

    examples = [
        example_basic_extraction,
        example_performance_extraction,
        example_custom_extraction,
        example_querying_data,
        example_custom_slot_groups,
        example_programmatic_usage,
        example_analyzing_fee_trends,
        example_complex_queries,
    ]

    if len(sys.argv) > 1:
        try:
            example_num = int(sys.argv[1])
            if 1 <= example_num <= len(examples):
                examples[example_num - 1]()
            else:
                print(f"Example {example_num} not found. Available: 1-{len(examples)}")
        except ValueError:
            print("Please provide a number between 1 and", len(examples))
    else:
        for i, example in enumerate(examples, 1):
            example()
            if i < len(examples):
                print("\n" + "=" * 60 + "\n")

    print("\nTo run a specific example:")
    print("python examples.py <example_number>")
    print(f"Available examples: 1-{len(examples)}")


if __name__ == "__main__":
    main()
