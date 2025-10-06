import pandas as pd
import ogmios
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from config import ExtractionConfig


def get_slot_group_directory(slot: int, group_size: int = 10000) -> str:
    """Get the directory name for a slot group."""
    group = slot // group_size
    return f"slot_{group * group_size}_{(group + 1) * group_size - 1}"


def extract_transaction_data(tx: Dict[str, Any], slot: int) -> Dict[str, Any]:
    """Extract relevant data from a transaction."""
    tx_id = tx.get("id", "")
    tx_fee = tx.get("fee", {}).get("ada", {}).get("lovelace", 0)

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

    return {
        "slot": slot,
        "tx_id": tx_id,
        "tx_fee": tx_fee,
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
        tqdm.write(
            f"Appended {len(df)} transactions to {parquet_file} (total: {len(combined_df)})"
        )
    else:
        df.to_parquet(parquet_file, index=False)
        tqdm.write(f"Created {parquet_file} with {len(df)} transactions")


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

    # Track transactions by slot group
    transactions_buffer = {}  # slot_group -> list of transactions
    last_buffer_flush_slot = 0  # last slot when all buffers were flushed

    def flush_all_buffers(current_slot):
        """Flush all transaction buffers to parquet files."""
        nonlocal last_buffer_flush_slot
        flushed_count = 0
        for slot_group_dir, txs in transactions_buffer.items():
            if txs:
                save_transactions_to_parquet(txs, slot_group_dir, config)
                flushed_count += len(txs)
                transactions_buffer[slot_group_dir] = []
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
                                tx_data = extract_transaction_data(tx, current_slot)
                                total_txs_processed += 1

                                # Add to buffer for this slot group
                                if slot_group_dir not in transactions_buffer:
                                    transactions_buffer[slot_group_dir] = []

                                transactions_buffer[slot_group_dir].append(tx_data)

                                # Check if we should flush all buffers based on slot difference
                                slots_since_last_flush = (
                                    current_slot - last_buffer_flush_slot
                                )
                                if slots_since_last_flush >= config.buffer_size_slots:
                                    flushed_count = flush_all_buffers(current_slot)
                                    transactions_buffer = {}
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
