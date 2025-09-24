# Cardano Fee Analytics with DuckDB

This project analyzes Cardano transaction fees by extracting blockchain data and storing it in Parquet files for efficient querying with DuckDB. The goal is to analyze the contribution of various Cardano projects to the network's sustainability by generating transaction fees.

## Overview

The system extracts transaction data from the Cardano blockchain and organizes it into Parquet files for analysis. Each transaction record includes:

- Transaction ID
- Transaction fees (in lovelace and ADA)
- Input UTXOs (transaction ID + output index)
- Output addresses
- Slot number and other metadata

## Data Storage Structure

Data is organized in the `duckdb/` directory with the following structure:

```
duckdb/
├── slot_0_9999/
│   └── tx.parquet
├── slot_10000_19999/
│   └── tx.parquet
├── slot_20000_29999/
│   └── tx.parquet
└── ...
```

Each directory contains transactions for a group of 10,000 slots, making it easy to query specific time ranges efficiently.

## Prerequisites

1. **Cardano Node**: You need a fully synced Cardano node running
2. **Ogmios**: Install and run Ogmios connected to your Cardano node
   ```bash
   # Install Ogmios (requires Node.js)
   npm install -g @cardano-ogmios/server
   
   # Run Ogmios (adjust paths as needed)
   ogmios --host 0.0.0.0 --port 1337 --node-socket /path/to/cardano-node/socket
   ```

## Installation and Setup

1. Clone this repository and navigate to it
2. Install dependencies:
   ```bash
   uv sync
   ```
3. Test the setup:
   ```bash
   python test_setup.py
   ```

## Usage

### Quick Start

1. **Basic extraction** (extracts all transactions from SNEK mint point):
   ```bash
   python main.py extract
   ```

2. **Query high-fee transactions** (after extraction):
   ```bash
   python main.py query
   ```

3. **View usage examples**:
   ```bash
   python examples.py
   ```

### Data Extraction Options

#### Basic Extraction
```bash
# Extract all transactions (default settings)
python main.py extract

# Extract from a specific era
python main.py extract --start-point shelley_era
python main.py extract --start-point alonzo_era
```

#### High-Fee Extraction
```bash
# Extract only transactions with fees > 2 ADA
python main.py extract --config high-fee

# Extract transactions with fees > 5 ADA
python main.py extract --min-fee 5.0
```

#### Custom Extraction
```bash
# Custom batch and buffer sizes for performance tuning
python main.py extract --batch-size 25 --buffer-size 2000

# Full blockchain history (warning: large dataset!)
python main.py extract --config full-history
```

#### Available Starting Points
- `shelley_era`: Beginning of Shelley era (slot 4492800)
- `mary_era`: Beginning of Mary era - multi-asset support (slot 16588800)
- `alonzo_era`: Beginning of Alonzo era - smart contracts (slot 39916975)
- `babbage_era`: Beginning of Babbage era - current era (slot 72316896)
- `snek_mint`: Block before SNEK token mint (slot 90914081) - **default**
- `recent`: Recent block for testing (slot 133660799)

### Data Analysis and Querying

#### Built-in Query Tools
```bash
# Query transactions with fees > 2 ADA
python query_high_fees.py

# Query transactions with fees > 5 ADA
python query_high_fees.py --min-fee 5.0

# Query transactions within specific slot range
python query_high_fees.py --slot-range 90914081 90920000

# Alternative: use main script
python main.py query
```

#### Direct DuckDB Queries

```python
import duckdb

conn = duckdb.connect()

# Find highest fee transactions
high_fee_txs = conn.execute("""
    SELECT tx_id, tx_fee_ada, slot, num_inputs, num_outputs
    FROM read_parquet('duckdb/*/tx.parquet')
    WHERE tx_fee_ada > 2.0
    ORDER BY tx_fee_ada DESC
    LIMIT 20
""").fetchdf()

# Analyze fee trends over time
fee_trends = conn.execute("""
    SELECT
        (slot // 4320) * 4320 as day_start_slot,
        COUNT(*) as daily_tx_count,
        AVG(tx_fee_ada) as avg_daily_fee,
        MAX(tx_fee_ada) as max_daily_fee
    FROM read_parquet('duckdb/*/tx.parquet')
    GROUP BY day_start_slot
    ORDER BY day_start_slot
""").fetchdf()

conn.close()
```

#### Programmatic Usage

```python
from config import ExtractionConfig, get_high_fee_config
from main import main

# Custom configuration
config = ExtractionConfig(
    min_fee_ada=1.0,
    batch_size=20,
    buffer_size=500,
    output_dir="custom_data"
)

# Run extraction
main(config)
```

## Data Schema

Each Parquet file contains the following columns:

- `slot`: The slot number when the transaction was included
- `tx_id`: Transaction hash/identifier
- `tx_fee_lovelace`: Transaction fee in lovelace
- `tx_fee_ada`: Transaction fee in ADA (lovelace / 1,000,000)
- `inputs`: Pipe-separated list of input UTXOs (format: `tx_id#output_index`)
- `output_addresses`: Pipe-separated list of output addresses
- `num_inputs`: Number of inputs in the transaction
- `num_outputs`: Number of outputs in the transaction

## Analysis Examples

The system enables various types of analysis:

1. **High-fee transaction identification**: Find transactions that paid unusually high fees
2. **Fee trend analysis**: Track how fees change over time
3. **Address-based analysis**: Analyze fee patterns for specific addresses
4. **Time-based analysis**: Examine fee patterns during specific periods


## Performance Considerations

- Data is processed in batches to improve throughput
- Parquet files are organized by slot groups for efficient time-based queries
- Transaction buffers prevent excessive I/O operations
- DuckDB enables fast analytical queries without loading all data into memory

## Advanced Usage

### Address-Focused Analysis

Extract transactions involving specific addresses:

```python
from config import get_address_focused_config, KNOWN_ADDRESSES
from main import main

# Use predefined addresses (DEX addresses, etc.)
target_addresses = [
    KNOWN_ADDRESSES["minswap_v1"],
    KNOWN_ADDRESSES["sundaeswap_v1"]
]

config = get_address_focused_config(target_addresses)
main(config)
```

### Performance Tuning

- **Batch size**: Increase for better throughput (10-50)
- **Buffer size**: Adjust based on available memory (500-5000)
- **Slot group size**: Larger groups for historical data analysis (default: 10,000)

### Monitoring Progress

The system provides progress updates showing:
- Blocks processed
- Current slot number
- Total transactions processed
- Files created and their locations

## Troubleshooting

### Common Issues

1. **Ogmios connection failed**: Ensure Ogmios is running and accessible
2. **Out of memory**: Reduce batch size and buffer size
3. **Disk space**: Monitor the `duckdb/` directory size
4. **Import errors**: Run `python test_setup.py` to verify setup

### Performance Tips

- Use SSD storage for better I/O performance
- Increase buffer sizes if you have sufficient RAM
- Use fee filtering to reduce dataset size
- Consider running extraction in stages (by era)

## Examples and Tutorials

Run the examples script to see detailed usage patterns:

```bash
# View all examples
python examples.py

# View specific example
python examples.py 1  # Basic extraction
python examples.py 4  # Querying data
python examples.py 7  # Fee trend analysis
```

## Future Enhancements

- Real-time data streaming for ongoing analysis
- Support for filtering by specific tokens/assets
- Address clustering and analysis
- Visualization dashboards for fee trends
- Integration with other Cardano analytics tools
- Support for custom fee calculation algorithms

## Dependencies

- `ogmios`: For blockchain data access
- `pandas`: For data manipulation
- `pyarrow`: For Parquet file support
- `duckdb`: For analytical queries
- `networkx`: For network analysis capabilities
