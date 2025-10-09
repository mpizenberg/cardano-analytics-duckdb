import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ogmios
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from config import ExtractionConfig


def get_parquet_schema(data_type: str) -> pa.Schema:
    """Get PyArrow schema for each parquet file type."""
    if data_type == "tx_raw":
        return pa.schema(
            [
                pa.field("tx_id", pa.binary(32)),
                pa.field("slot", pa.uint64()),
                pa.field("raw_cbor", pa.binary()),
            ]
        )

    elif data_type == "tx":
        return pa.schema(
            [
                pa.field("slot", pa.uint64()),
                pa.field("tx_id", pa.binary(32)),
                pa.field("tx_fee", pa.uint64()),
                pa.field("input_count", pa.uint16()),
                pa.field("output_count", pa.uint16()),
                pa.field("redeemer_count", pa.uint16()),
                pa.field("has_mint", pa.bool_()),
                pa.field("has_withdrawal", pa.bool_()),
                pa.field("has_cert", pa.bool_()),
                pa.field("has_vote", pa.bool_()),
                pa.field("has_proposal", pa.bool_()),
            ]
        )

    elif data_type == "utxo":
        return pa.schema(
            [
                pa.field("slot", pa.uint64()),
                pa.field("tx_id", pa.binary(32)),
                pa.field("output_index", pa.uint16()),
                pa.field("address", pa.dictionary(pa.int32(), pa.string())),
                pa.field("lovelace", pa.uint64()),
                pa.field("has_token", pa.bool_()),
                pa.field("has_datum", pa.bool_()),
                pa.field("has_ref_script", pa.bool_()),
            ]
        )

    elif data_type == "mint":
        return pa.schema(
            [
                pa.field("slot", pa.uint64()),
                pa.field("tx_id", pa.binary(32)),
                pa.field("policy_id", pa.binary(28)),
                pa.field("asset_name", pa.binary()),
                pa.field("quantity", pa.int64()),
            ]
        )

    elif data_type == "cert":
        return pa.schema(
            [
                pa.field("slot", pa.uint64()),
                pa.field("tx_id", pa.binary(32)),
                pa.field("index", pa.uint16()),
                pa.field("type", pa.dictionary(pa.int8(), pa.string())),
            ]
        )

    else:
        raise ValueError(f"Unknown data type: {data_type}")


def get_compression_config(data_type: str) -> dict:
    """Get optimal compression configuration for each file type."""
    compression_configs = {
        "tx_raw": {
            "compression": "brotli",
            "compression_level": 6,
            "use_dictionary": True,
        },
        "utxo": {
            "compression": "brotli",
            "compression_level": 6,
            "use_dictionary": True,
        },
    }

    return compression_configs.get(
        data_type,
        {},
    )


def get_slot_group_directory(slot: int, group_size: int = 10000) -> str:
    """Get the directory name for a slot group."""
    group = slot // group_size
    return f"slot_{group * group_size}_{(group + 1) * group_size - 1}"


def extract_transaction_raw_data(tx: Dict[str, Any], slot: int) -> Dict[str, Any]:
    """Extract raw transaction data for tx-raw.parquet file."""
    # Store the full transaction data with minimal processing
    return {
        "tx_id": bytes.fromhex(tx.get("id", "0" * 64)),
        "slot": slot,
        "raw_cbor": bytes.fromhex(tx.get("cbor", "")),
    }


def extract_transaction_data(tx: Dict[str, Any], slot: int) -> Dict[str, Any]:
    """Extract relevant data for tx.parquet file."""
    # TODO: block height
    # TODO: ref inputs
    return {
        "slot": slot,
        "tx_id": bytes.fromhex(tx.get("id", "0" * 64)),
        "tx_fee": tx.get("fee", {}).get("ada", {}).get("lovelace", 0),
        "input_count": len(tx.get("inputs", [])),
        "output_count": len(tx.get("outputs", [])),
        "redeemer_count": len(tx.get("redeemers", [])),
        "has_mint": bool(tx.get("mint", [])),
        "has_withdrawal": bool(tx.get("withdrawals", [])),
        "has_cert": bool(tx.get("certificates", [])),
        "has_vote": bool(tx.get("votes", [])),
        "has_proposal": bool(tx.get("proposals", [])),
    }


def extract_utxo_data(tx: Dict[str, Any], slot: int) -> List[Dict[str, Any]]:
    """Extract UTxO data for utxo.parquet file."""
    utxos = []
    # TODO: `is_script_address`: BOOLEAN
    if tx.get("outputs"):
        for i, output in enumerate(tx["outputs"]):
            value = output.get("value", {})
            address = output.get("address", "")
            utxo = {
                "slot": slot,
                "tx_id": bytes.fromhex(tx.get("id", "0" * 64)),
                "output_index": i,
                "address": address,
                "lovelace": value.get("ada", {}).get("lovelace", 0),
                "has_token": len(value) > 1,
                "has_datum": bool(output.get("datum")),
                "has_ref_script": bool(output.get("script")),
            }
            utxos.append(utxo)

    return utxos


def extract_mint_data(tx: Dict[str, Any], slot: int) -> List[Dict[str, Any]]:
    """Extract minting data for mint.parquet file."""
    mint_records = []
    if tx.get("mint"):
        for policy_id, assets in tx["mint"].items():
            for asset_name, quantity in assets.items():
                mint_record = {
                    "slot": slot,
                    "tx_id": bytes.fromhex(tx.get("id", "0" * 64)),
                    "policy_id": bytes.fromhex(policy_id),
                    "asset_name": bytes.fromhex(asset_name),
                    "quantity": quantity,
                }
                mint_records.append(mint_record)

    return mint_records


def extract_certificate_data(tx: Dict[str, Any], slot: int) -> List[Dict[str, Any]]:
    """Extract certificate data for cert.parquet file."""
    certificates = []
    if tx.get("certificates"):
        for i, cert in enumerate(tx["certificates"]):
            cert_type = cert.get("type", "unknown")
            cert_data = {
                "slot": slot,
                "tx_id": bytes.fromhex(tx.get("id", "0" * 64)),
                "index": i,
                "type": cert_type,  # TODO: convert to int
            }
            certificates.append(cert_data)

    return certificates


def save_parquet_with_schema(df: pd.DataFrame, file_path: Path, data_type: str):
    """Save DataFrame to parquet with proper schema and compression."""
    if df.empty:
        return

    # Get schema and compression config for this data type
    schema = get_parquet_schema(data_type)
    compression_config = get_compression_config(data_type)

    # Convert DataFrame to PyArrow table with proper schema
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)

    # Write parquet file with compression settings
    pq.write_table(table, file_path, **compression_config)


def save_to_parquet(
    data: List[Dict[str, Any]],
    file_name: str,
    slot_group_dir: str,
    config: ExtractionConfig,
    dedup_cols: Optional[List[str]] = None,
):
    """
    Save data to a parquet file in the appropriate slot group directory.

    Args:
        data: List of dictionaries containing data to save
        file_name: Name of the parquet file (e.g., "tx.parquet", "utxo.parquet")
        slot_group_dir: Directory name for the slot group
        config: Extraction configuration
        dedup_cols: Column(s) to use for deduplication (None = no deduplication)
    """
    if not data:
        return 0

    # Create the full directory path
    duckdb_dir = Path(config.output_dir)
    group_dir = duckdb_dir / slot_group_dir
    group_dir.mkdir(parents=True, exist_ok=True)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Determine data type from filename
    data_type = file_name.split(".")[0].replace("-", "_")  # tx-raw.parquet -> tx_raw
    parquet_file = group_dir / file_name

    # If file exists, append to it, otherwise create new
    if parquet_file.exists():
        try:
            existing_df = pd.read_parquet(parquet_file)
            combined_df = pd.concat([existing_df, df], ignore_index=True)

            # Remove duplicates if dedup columns are specified
            if dedup_cols:
                pre_dedup_count = len(combined_df)
                combined_df = combined_df.drop_duplicates(
                    subset=dedup_cols, keep="last"
                )
                dedup_count = pre_dedup_count - len(combined_df)
                dedup_msg = (
                    f", removed {dedup_count} duplicates" if dedup_count > 0 else ""
                )
            else:
                dedup_msg = ""

            save_parquet_with_schema(combined_df, parquet_file, data_type)
            tqdm.write(
                f"Appended {len(df)} records to {parquet_file} (total: {len(combined_df)}{dedup_msg})"
            )
            return len(df)
        except Exception as e:
            tqdm.write(f"Error appending to {parquet_file}: {e}")
            # If there's an error with the existing file, create a new one
            save_parquet_with_schema(df, parquet_file, data_type)
            tqdm.write(f"Created new {parquet_file} with {len(df)} records")
            return len(df)
    else:
        save_parquet_with_schema(df, parquet_file, data_type)
        tqdm.write(f"Created {parquet_file} with {len(df)} records")
        return len(df)


def extract_transactions(config: ExtractionConfig):
    """
    Main function to extract transaction data from Cardano blockchain using Ogmios.

    This function connects to an Ogmios client, processes blocks, and saves transaction
    data to parquet files organized by slot groups.

    Args:
        config: An ExtractionConfig object containing extraction parameters
    """
    tqdm.write("Starting Cardano fee analytics data extraction...")
    tqdm.write(
        f"Configuration: batch_size={config.batch_size}, buffer_size_slots={config.buffer_size_slots}"
    )
    tqdm.write(f"Output directory: {config.output_dir}")
    tqdm.write(f"Slot group size: {config.slot_group_size}")

    # Track data by slot group, with each group containing different data types
    data_buffers = {}  # slot_group -> {"tx": [], "tx_raw": [], ...}

    last_buffer_flush_slot = 0  # last slot when all buffers were flushed

    def flush_all_buffers(current_slot):
        """Flush all data buffers to parquet files."""
        nonlocal last_buffer_flush_slot
        flushed_count = 0

        # Define file configurations: file name and deduplication columns
        file_configs = {
            "tx_raw": {"file": "tx-raw.parquet", "dedup": ["tx_id"]},
            "tx": {"file": "tx.parquet", "dedup": ["tx_id"]},
            "utxo": {"file": "utxo.parquet", "dedup": ["tx_id", "output_index"]},
            "mint": {
                "file": "mint.parquet",
                "dedup": ["tx_id", "policy_id", "asset_name"],
            },
            "cert": {"file": "cert.parquet", "dedup": ["tx_id", "index"]},
            # "vote": {"file": "vote.parquet", "dedup": ["tx_id", "index"]},
            # "redeemer": {"file": "redeemer.parquet", "dedup": ["tx_id", "index"]},
            # "proposal": {"file": "proposal.parquet", "dedup": ["tx_id", "index"]},
        }

        # Process each slot group
        for slot_group_dir, data_types in data_buffers.items():
            # Process each data type in this slot group
            for data_type, buffer in data_types.items():
                if buffer and data_type in file_configs:
                    config_data = file_configs[data_type]
                    flushed_count += save_to_parquet(
                        data=buffer,
                        file_name=config_data["file"],
                        slot_group_dir=slot_group_dir,
                        config=config,
                        dedup_cols=config_data["dedup"],
                    )
                    # Clear the buffer
                    data_buffers[slot_group_dir][data_type] = []

        last_buffer_flush_slot = current_slot
        return flushed_count

    with ogmios.Client(host=config.ogmios_host, port=config.ogmios_port) as client:
        tqdm.write("Connected to Ogmios client")

        # Set chain pointer based on configuration
        if config.start_point:
            point, tip, _ = client.find_intersection.execute([config.start_point])
            tqdm.write(f"Starting from slot {point.slot}")
            start_slot = point.slot
        else:
            tqdm.write("No starting point specified, using chain tip")
            point, tip, _ = client.find_intersection.execute([ogmios.Origin()])
            start_slot = tip.slot

        # Calculate total slots for progress tracking
        stop_slot = (
            min(config.stop_point.slot, tip.slot) if config.stop_point else tip.slot
        )
        total_slots = stop_slot - start_slot if start_slot else stop_slot

        # Initialize progress bar
        pbar = tqdm(
            total=total_slots,
            desc="Processing blocks",
            unit="slots",
            position=0,
            leave=True,
            bar_format="{desc}: {percentage:3.1f}%|{bar}| {n:,}/{total:,} slots [{elapsed}<{remaining}, {rate_fmt}]",
        )

        total_txs_processed = 0
        blocks_processed = 0
        current_slot = start_slot

        try:
            while True:
                # Batch requests to improve performance
                for i in range(config.batch_size):
                    client.next_block.send()

                for i in range(config.batch_size):
                    direction, tip, block, id = client.next_block.receive()
                    if direction.value == "forward":
                        blocks_processed += 1
                        current_slot = getattr(block, "slot", 0)

                        # Process transactions in the block
                        if isinstance(block, ogmios.Block) and hasattr(
                            block, "transactions"
                        ):
                            slot_group_dir = get_slot_group_directory(
                                current_slot, config.slot_group_size
                            )

                            for tx in block.transactions:
                                # Process basic transaction data (tx.parquet)
                                tx_data = extract_transaction_data(tx, current_slot)
                                total_txs_processed += 1

                                # Process raw transaction data (tx-raw.parquet)
                                tx_raw_data = extract_transaction_raw_data(
                                    tx, current_slot
                                )

                                # Process UTxO data (utxo.parquet)
                                utxo_data = extract_utxo_data(tx, current_slot)

                                # Process mint data (mint.parquet)
                                mint_data = extract_mint_data(tx, current_slot)

                                # Process certificate data (cert.parquet)
                                cert_data = extract_certificate_data(tx, current_slot)

                                # Initialize slot group buffer if it doesn't exist
                                if slot_group_dir not in data_buffers:
                                    data_buffers[slot_group_dir] = {
                                        "tx": [],
                                        "tx_raw": [],
                                        "utxo": [],
                                        "mint": [],
                                        "cert": [],
                                        # "vote": [],
                                        # "redeemer": [],
                                        # "proposal": [],
                                    }

                                # Add to respective buffers for this slot group
                                data_buffers[slot_group_dir]["tx"].append(tx_data)
                                data_buffers[slot_group_dir]["tx_raw"].append(
                                    tx_raw_data
                                )

                                # Add data that might be empty
                                if utxo_data:
                                    data_buffers[slot_group_dir]["utxo"].extend(
                                        utxo_data
                                    )
                                if mint_data:
                                    data_buffers[slot_group_dir]["mint"].extend(
                                        mint_data
                                    )
                                if cert_data:
                                    data_buffers[slot_group_dir]["cert"].extend(
                                        cert_data
                                    )

                                # Check if we should flush all buffers based on slot difference
                                slots_since_last_flush = (
                                    current_slot - last_buffer_flush_slot
                                )
                                if slots_since_last_flush >= config.buffer_size_slots:
                                    flushed_count = flush_all_buffers(current_slot)
                                    # Reset all data buffers
                                    data_buffers = {}
                                    # Update progress bar only when flushing buffers
                                    slots_progress = current_slot - start_slot
                                    pbar.n = min(slots_progress, total_slots)
                                    pbar.set_postfix(
                                        {
                                            "blocks": f"{blocks_processed:,}",
                                            "txs": f"{total_txs_processed:,}",
                                            "slot": f"{current_slot:,}",
                                            "flushed": f"{flushed_count:,}",
                                        }
                                    )
                                    pbar.refresh()

                        # Stop when we've reached the network tip
                        stop_point = (
                            min(config.stop_point.slot, tip.slot)
                            if config.stop_point
                            else tip.slot
                        )
                        if block.slot >= stop_point:
                            # Final progress bar update
                            slots_progress = current_slot - start_slot
                            pbar.n = min(slots_progress, total_slots)
                            pbar.set_postfix(
                                {
                                    "blocks": f"{blocks_processed:,}",
                                    "txs": f"{total_txs_processed:,}",
                                    "slot": f"{current_slot:,}",
                                }
                            )
                            pbar.refresh()
                            pbar.close()
                            if tip.height == block.height:
                                tqdm.write(f"Reached chain tip at slot {tip.slot}")
                            else:
                                tqdm.write(f"Reached stop point at slot {stop_point}")
                            # Save any remaining transactions in buffers
                            remaining_count = flush_all_buffers(current_slot)
                            if remaining_count > 0:
                                tqdm.write(
                                    f"Flushed {remaining_count} remaining transactions at stop point"
                                )

                            tqdm.write(
                                f"Data extraction complete. Total transactions processed: {total_txs_processed:,}"
                            )
                            tqdm.write(
                                f"Data saved in {config.output_dir}/ directory organized by slot groups"
                            )
                            return

                    elif direction.value == "backward":
                        tqdm.write("Encountered rollback, continuing...")
        finally:
            # Ensure progress bar is closed even if an exception occurs
            pbar.close()
