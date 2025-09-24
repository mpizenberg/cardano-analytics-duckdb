#!/usr/bin/env python3
"""
Test script to validate the fee analytics setup.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        import pandas as pd

        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import pandas: {e}")
        return False

    try:
        import pyarrow

        print("✓ pyarrow imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import pyarrow: {e}")
        return False

    try:
        import duckdb

        print("✓ duckdb imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import duckdb: {e}")
        return False

    try:
        import ogmios

        print("✓ ogmios imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ogmios: {e}")
        return False

    try:
        from config import ExtractionConfig, get_default_config

        print("✓ config module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import config: {e}")
        return False

    return True


def test_directory_structure():
    """Test that required directories can be created."""
    print("\nTesting directory structure...")

    test_dir = Path("test_duckdb")
    slot_dir = test_dir / "slot_0_9999"

    try:
        slot_dir.mkdir(parents=True, exist_ok=True)
        print("✓ Created test directory structure")

        # Clean up
        import shutil

        shutil.rmtree(test_dir)
        print("✓ Cleaned up test directories")
        return True
    except Exception as e:
        print(f"✗ Failed to create directories: {e}")
        return False


def test_parquet_operations():
    """Test basic parquet file operations."""
    print("\nTesting parquet operations...")

    try:
        import pandas as pd

        # Create sample transaction data
        sample_data = [
            {
                "slot": 90914100,
                "tx_id": "test123",
                "tx_fee_lovelace": 200000,
                "tx_fee_ada": 0.2,
                "inputs": "input1#0|input2#1",
                "output_addresses": "addr1|addr2",
                "num_inputs": 2,
                "num_outputs": 2,
            }
        ]

        df = pd.DataFrame(sample_data)

        # Test parquet write/read
        test_file = Path("test_tx.parquet")
        df.to_parquet(test_file, index=False)
        print("✓ Created test parquet file")

        # Read it back
        df_read = pd.read_parquet(test_file)
        assert len(df_read) == 1
        assert df_read.iloc[0]["tx_id"] == "test123"
        print("✓ Read test parquet file successfully")

        # Clean up
        test_file.unlink()
        print("✓ Cleaned up test parquet file")
        return True

    except Exception as e:
        print(f"✗ Parquet operations failed: {e}")
        return False


def test_duckdb_query():
    """Test basic DuckDB operations."""
    print("\nTesting DuckDB operations...")

    try:
        import duckdb
        import pandas as pd

        # Create sample data and save to parquet
        sample_data = [
            {"tx_id": "tx1", "tx_fee_ada": 0.5, "slot": 1000},
            {"tx_id": "tx2", "tx_fee_ada": 3.0, "slot": 1001},
            {"tx_id": "tx3", "tx_fee_ada": 1.5, "slot": 1002},
        ]

        test_dir = Path("test_query")
        test_dir.mkdir(exist_ok=True)
        test_file = test_dir / "test.parquet"

        df = pd.DataFrame(sample_data)
        df.to_parquet(test_file, index=False)

        # Test DuckDB query
        conn = duckdb.connect()
        result = conn.execute(f"""
            SELECT COUNT(*) as high_fee_count
            FROM read_parquet('{test_file}')
            WHERE tx_fee_ada > 2.0
        """).fetchone()

        assert result[0] == 1, f"Expected 1 high-fee transaction, got {result[0]}"
        print("✓ DuckDB query executed successfully")

        conn.close()

        # Clean up
        import shutil

        shutil.rmtree(test_dir)
        print("✓ Cleaned up test query files")
        return True

    except Exception as e:
        print(f"✗ DuckDB operations failed: {e}")
        return False


def test_configuration():
    """Test configuration system."""
    print("\nTesting configuration system...")

    try:
        from config import (
            get_default_config,
            get_high_fee_config,
            PRESET_STARTING_POINTS,
        )

        # Test default config
        config = get_default_config()
        assert config.batch_size == 10
        assert config.buffer_size == 1000
        print("✓ Default configuration loaded")

        # Test high-fee config
        high_fee_config = get_high_fee_config(5.0)
        assert high_fee_config.min_fee_ada == 5.0
        print("✓ High-fee configuration loaded")

        # Test preset starting points
        assert "snek_mint" in PRESET_STARTING_POINTS
        print("✓ Preset starting points available")

        return True

    except Exception as e:
        print(f"✗ Configuration system failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Fee Analytics Setup Test")
    print("=" * 40)

    tests = [
        test_imports,
        test_directory_structure,
        test_parquet_operations,
        test_duckdb_query,
        test_configuration,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"✗ {test.__name__} failed")
        except Exception as e:
            print(f"✗ {test.__name__} crashed: {e}")

    print("\n" + "=" * 40)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Setup is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check the setup.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
