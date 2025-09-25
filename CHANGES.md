# Changes Made to Cardano Fee Analytics System

This document summarizes the significant changes made to improve the fee analytics system based on the requirements.

## Configuration System Changes

### Removed Filtering Options
- **Removed**: `min_fee_ada` filtering from `ExtractionConfig`
- **Removed**: `target_addresses` filtering from `ExtractionConfig`  
- **Removed**: `target_assets` filtering from `ExtractionConfig`
- **Removed**: `max_file_size_mb` configuration option
- **Rationale**: Filtering should be done at query time, not during extraction, to maintain data completeness

### Buffer and Slot Group Measurement
- **Changed**: `buffer_size` (transaction count) → `buffer_size_slots` (slot count)
- **Ensured**: Both `buffer_size_slots` and `slot_group_size` use the same unit (slots)
- **Rationale**: Prevents temporal inconsistencies where buffer might span longer than slot groups

### Starting Points Correction
Updated all starting points to be the **last block before** each era, not the first block of each era:

- `last_byron`: slot 4492799, hash f8084c61b6a238acec985b59310b6ecec49c0ab8352249afd7268da5cff2a457
- `last_shelley`: slot 16588737, hash 4e9bbbb67e3ae262133d94c3da5bffce7b1127fc436e7433b87668dba34c354a
- `last_allegra`: slot 23068793, hash 69c44ac1dda2ec74646e4223bc804d9126f719b1c245dadc2ad65e8de1b276d7
- `last_mary`: slot 39916796, hash e72579ff89dc9ed325b723a33624b596c08141c7bd573ecfff56a1f7229e4d09
- `last_alonzo`: slot 72316796, hash c58a24ba8203e7629422a24d9dc68ce2ed495420bf40d9dab124373655161a20
- `last_babbage`: slot 133660799, hash e757d57eb8dc9500a61c60a39fadb63d9be6973ba96ae337fd24453d4d15c343

**Rationale**: Chain sync gives the next block after intersection, so using the last block before an era ensures we start from the beginning of the target era.

## Main Script Changes

### Extraction Logic
- **Simplified**: `extract_transaction_data()` function - removed filtering logic
- **Updated**: Buffer management to use slot-based timing instead of transaction counting
- **Improved**: Buffer saving now triggers based on `slots_since_last_save >= config.buffer_size_slots`

### Configuration Presets
- **Renamed**: `get_high_fee_config()` → `get_performance_config()`
- **Removed**: `get_address_focused_config()` function
- **Updated**: All preset configs to use `buffer_size_slots` instead of `buffer_size`

### Command Line Interface
- **Removed**: `--min-fee` argument
- **Changed**: `--buffer-size` → `--buffer-size-slots`
- **Added**: `--slot-group-size` argument
- **Updated**: Configuration choices from `high-fee` to `performance`

## File Updates

### config.py
- Complete restructure of `ExtractionConfig` dataclass
- Updated all preset starting points with correct slot numbers and hashes
- Removed filtering-related configuration options
- Changed buffer measurement from transactions to slots

### main.py
- Removed all filtering logic from extraction process
- Updated buffer management to be slot-based
- Simplified transaction data extraction
- Updated command line argument parsing

### query_high_fees.py
- Minor formatting fixes
- Maintained all querying functionality (filtering is now done at query time)

### examples.py
- Updated all examples to reflect configuration changes
- Removed address-focused analysis example
- Added custom slot group configuration example
- Updated starting point names throughout

### test_setup.py
- Updated configuration tests to match new structure
- Added tests for new starting points

### README.md
- Complete rewrite of usage sections
- Updated starting point documentation
- Removed filtering-related documentation
- Added performance tuning guidelines
- Updated all code examples

## Benefits of Changes

### Improved Data Consistency
- All transactions are now extracted, maintaining complete blockchain data
- Filtering at query time allows for multiple analysis perspectives on the same dataset

### Better Temporal Control
- Slot-based buffer management provides consistent temporal granularity
- Prevents buffer overflow beyond slot group boundaries
- More predictable file creation timing

### Correct Chain Synchronization
- Starting points now correctly position for era-based analysis
- Chain sync will properly deliver blocks from the beginning of target eras

### Enhanced Usability
- Clearer configuration options focused on performance rather than filtering
- More intuitive slot-based measurements
- Better alignment between buffer size and slot group size

## Migration Guide

If you were using the old system:

1. **Replace filtering configurations**:
   ```python
   # Old
   config = get_high_fee_config(min_fee_ada=2.0)
   
   # New - extract all, filter at query time
   config = get_performance_config()
   ```

2. **Update buffer configuration**:
   ```python
   # Old
   config.buffer_size = 1000  # transactions
   
   # New
   config.buffer_size_slots = 1000  # slots
   ```

3. **Update starting points**:
   ```python
   # Old
   config.start_point = PRESET_STARTING_POINTS["shelley_era"]
   
   # New
   config.start_point = PRESET_STARTING_POINTS["last_byron"]
   ```

4. **Apply filters at query time**:
   ```sql
   -- Query high-fee transactions
   SELECT * FROM read_parquet('duckdb/*/tx.parquet')
   WHERE tx_fee_ada > 2.0
   ```

## Testing

All changes have been validated with the test suite:
- Configuration system tests pass
- Parquet operations work correctly
- DuckDB queries function as expected
- Import dependencies are satisfied

Run `python test_setup.py` to verify your installation after these changes.