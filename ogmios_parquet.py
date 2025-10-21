import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ogmios
from pathlib import Path
from typing import List, Dict, Any, Set
from tqdm import tqdm
from config import ExtractionConfig
import uuid
from collections import defaultdict


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
                pa.field(
                    "inputs",
                    pa.list_(
                        pa.struct(
                            [
                                pa.field("tx_id", pa.binary(32)),
                                pa.field("output_index", pa.uint16()),
                            ]
                        )
                    ),
                ),
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

    elif data_type == "asset":
        return pa.schema(
            [
                pa.field("slot", pa.uint64()),
                pa.field("tx_id", pa.binary(32)),
                pa.field("output_index", pa.uint16()),
                pa.field("address", pa.dictionary(pa.int32(), pa.string())),
                pa.field("policy_id", pa.binary(28)),
                pa.field("asset_name", pa.binary()),
                pa.field("amount", pa.uint64()),
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
            "compression_level": 4,
            "use_dictionary": True,
        },
        "tx": {
            "compression": "brotli",
            "compression_level": 4,
            "use_dictionary": True,
        },
        "utxo": {
            "compression": "brotli",
            "compression_level": 4,
            "use_dictionary": True,
        },
        "asset": {
            "compression": "brotli",
            "compression_level": 4,
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
    return {
        "tx_id": bytes.fromhex(tx.get("id", "0" * 64)),
        "slot": slot,
        "raw_cbor": bytes.fromhex(tx.get("cbor", "")),
    }


def extract_transaction_data(tx: Dict[str, Any], slot: int) -> Dict[str, Any]:
    """Extract relevant data for tx.parquet file."""
    # Extract input references
    inputs = []
    if tx.get("inputs"):
        for input_utxo in tx["inputs"]:
            # Input format from Ogmios: {"transaction": {"id": "..."}, "index": 0}
            if "transaction" in input_utxo and "id" in input_utxo["transaction"]:
                tx_id = input_utxo["transaction"]["id"]
                output_index = input_utxo.get("index", 0)
                inputs.append(
                    {"tx_id": bytes.fromhex(tx_id), "output_index": output_index}
                )

    # TODO: block height
    # TODO: ref inputs
    return {
        "slot": slot,
        "tx_id": bytes.fromhex(tx.get("id", "0" * 64)),
        "tx_fee": tx.get("fee", {}).get("ada", {}).get("lovelace", 0),
        "input_count": len(tx.get("inputs", [])),
        "inputs": inputs,
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


def extract_asset_data(tx: Dict[str, Any], slot: int) -> List[Dict[str, Any]]:
    """Extract asset data for asset.parquet file."""
    asset_records = []
    if tx.get("outputs"):
        for output_index, output in enumerate(tx["outputs"]):
            value = output.get("value", {})
            address = output.get("address", "")
            # Skip if no assets (only ADA)
            if len(value) <= 1:
                continue

            # Process each asset in the output
            for policy_id, assets in value.items():
                if policy_id == "ada":  # Skip ADA
                    continue
                for asset_name, amount in assets.items():
                    asset_record = {
                        "slot": slot,
                        "tx_id": bytes.fromhex(tx.get("id", "0" * 64)),
                        "output_index": output_index,
                        "address": address,
                        "policy_id": bytes.fromhex(policy_id),
                        "asset_name": bytes.fromhex(asset_name),
                        "amount": amount,
                    }
                    asset_records.append(asset_record)

    return asset_records


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
                "type": cert_type,
            }
            certificates.append(cert_data)

    return certificates


def save_to_parquet_uncompressed(
    data: List[Dict[str, Any]],
    file_name: str,
    slot_group_dir: str,
    config: ExtractionConfig,
    created_files: Set[Path],
):
    """Save data to a new, uncompressed parquet file for later merging."""
    if not data:
        return 0

    # Create the full directory path
    duckdb_dir = Path(config.output_dir)
    group_dir = duckdb_dir / slot_group_dir
    group_dir.mkdir(parents=True, exist_ok=True)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Create a unique filename for this chunk
    chunk_id = uuid.uuid4().hex
    base_name = Path(file_name).stem  # e.g. tx-raw
    temp_file_name = f"{base_name}_{chunk_id}.parquet"
    temp_parquet_file = group_dir / temp_file_name

    # Track this new file for merging and compression later
    created_files.add(temp_parquet_file)

    # Write uncompressed for speed
    df.to_parquet(temp_parquet_file, compression=None)
    tqdm.write(f"Wrote {len(df)} records to temporary file {temp_parquet_file}")
    return len(df)


def compress_final_files(created_files: Set[Path]):
    """Merge, compress, and save final parquet files."""
    tqdm.write(
        f"Starting final merge and compression of {len(created_files)} temporary files..."
    )

    # Group files by final destination file
    files_to_merge = defaultdict(list)
    for temp_file_path in created_files:
        base_name = temp_file_path.stem.rsplit("_", 1)[0]
        final_file_name = f"{base_name}.parquet"
        final_file_path = temp_file_path.parent / final_file_name
        files_to_merge[final_file_path].append(temp_file_path)

    # Process each group
    for final_file_path, chunk_files in tqdm(
        files_to_merge.items(), desc="Merging and compressing files"
    ):
        if not chunk_files:
            continue
        try:
            # Determine data type from filename
            data_type = final_file_path.stem.replace("-", "_")

            # Read all chunk files into a list of DataFrames
            df_list = [pd.read_parquet(f) for f in chunk_files]

            # Concatenate into a single DataFrame
            combined_df = pd.concat(df_list, ignore_index=True)

            # Write compressed with PyArrow schema
            schema = get_parquet_schema(data_type)
            compression_config = get_compression_config(data_type)
            table = pa.Table.from_pandas(
                combined_df, schema=schema, preserve_index=False
            )
            pq.write_table(table, final_file_path, **compression_config)
            tqdm.write(
                f"Created final file {final_file_path} with {len(chunk_files)} records."
            )

            # Clean up temporary chunk files
            for f in chunk_files:
                f.unlink()

        except Exception as e:
            tqdm.write(f"Error processing {final_file_path}: {e}")

    tqdm.write("Final merge and compression complete!")


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

    # Track data by slot group
    data_buffers = {}  # slot_group -> {"tx": [], "tx_raw": [], ...}
    created_files: Set[Path] = set()  # Track files for final compression
    last_buffer_flush_slot = 0

    def flush_all_buffers(current_slot):
        """Flush all data buffers to uncompressed parquet files."""
        nonlocal last_buffer_flush_slot
        flushed_count = 0

        # Define file configurations (no dedup since no duplicates exist)
        file_configs = {
            "tx_raw": {"file": "tx-raw.parquet"},
            "tx": {"file": "tx.parquet"},
            "utxo": {"file": "utxo.parquet"},
            "mint": {"file": "mint.parquet"},
            "asset": {"file": "asset.parquet"},
            "cert": {"file": "cert.parquet"},
        }

        # Process each slot group
        for slot_group_dir, data_types in data_buffers.items():
            # Process each data type in this slot group
            for data_type, buffer in data_types.items():
                if buffer and data_type in file_configs:
                    config_data = file_configs[data_type]
                    flushed_count += save_to_parquet_uncompressed(
                        data=buffer,
                        file_name=config_data["file"],
                        slot_group_dir=slot_group_dir,
                        config=config,
                        created_files=created_files,
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
                                # Process all data types
                                tx_data = extract_transaction_data(tx, current_slot)
                                total_txs_processed += 1

                                # tx_raw_data = extract_transaction_raw_data(
                                #     tx, current_slot
                                # )
                                utxo_data = extract_utxo_data(tx, current_slot)
                                mint_data = extract_mint_data(tx, current_slot)
                                asset_data = extract_asset_data(tx, current_slot)
                                cert_data = extract_certificate_data(tx, current_slot)

                                # Initialize slot group buffer if it doesn't exist
                                if slot_group_dir not in data_buffers:
                                    data_buffers[slot_group_dir] = {
                                        "tx": [],
                                        "tx_raw": [],
                                        "utxo": [],
                                        "mint": [],
                                        "asset": [],
                                        "cert": [],
                                    }

                                # Add to respective buffers
                                data_buffers[slot_group_dir]["tx"].append(tx_data)
                                # data_buffers[slot_group_dir]["tx_raw"].append(
                                #     tx_raw_data
                                # )

                                if utxo_data:
                                    data_buffers[slot_group_dir]["utxo"].extend(
                                        utxo_data
                                    )
                                if mint_data:
                                    data_buffers[slot_group_dir]["mint"].extend(
                                        mint_data
                                    )
                                if asset_data:
                                    data_buffers[slot_group_dir]["asset"].extend(
                                        asset_data
                                    )
                                if cert_data:
                                    data_buffers[slot_group_dir]["cert"].extend(
                                        cert_data
                                    )

                                # Flush buffers when needed
                                slots_since_last_flush = (
                                    current_slot - last_buffer_flush_slot
                                )
                                if slots_since_last_flush >= config.buffer_size_slots:
                                    flushed_count = flush_all_buffers(current_slot)
                                    # Reset all data buffers
                                    data_buffers = {}
                                    # Update progress bar
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

                        # Check stop condition
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
                                    f"Flushed {remaining_count} remaining transactions"
                                )

                            # Final compression phase
                            tqdm.write(
                                "Data extraction complete. Starting final compression..."
                            )
                            compress_final_files(created_files)

                            tqdm.write(
                                f"Simple optimized extraction complete. Total transactions: {total_txs_processed:,}"
                            )
                            tqdm.write(f"Data saved in {config.output_dir}/ directory")
                            return

                    elif direction.value == "backward":
                        tqdm.write("Encountered rollback, continuing...")

        finally:
            # Ensure progress bar is closed even if an exception occurs
            pbar.close()
