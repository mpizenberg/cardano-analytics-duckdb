import ogmios
from typing import Optional
from dataclasses import dataclass


@dataclass
class ExtractionConfig:
    """Configuration settings for blockchain data extraction."""

    # Ogmios connection settings
    ogmios_host: str = "localhost"
    ogmios_port: int = 1337

    # Starting and stopping point for extraction
    start_point: Optional[ogmios.Point] = None
    stop_point: Optional[ogmios.Point] = None

    # Batch processing settings
    batch_size: int = 200
    buffer_size_slots: int = (
        20000  # Save to parquet every N slots (flushes all buffers)
    )

    # Slot grouping (slots per directory)
    slot_group_size: int = 200000

    # Output directory
    output_dir: str = "duckdb"

    # Progress reporting
    progress_interval: int = 1000  # Report progress every N blocks


# Predefined starting points - last block before each era
PRESET_STARTING_POINTS = {
    "last_byron": ogmios.Point(
        slot=4492799,
        id="f8084c61b6a238acec985b59310b6ecec49c0ab8352249afd7268da5cff2a457",
    ),
    "last_shelley": ogmios.Point(
        slot=16588737,
        id="4e9bbbb67e3ae262133d94c3da5bffce7b1127fc436e7433b87668dba34c354a",
    ),
    "last_allegra": ogmios.Point(
        slot=23068793,
        id="69c44ac1dda2ec74646e4223bc804d9126f719b1c245dadc2ad65e8de1b276d7",
    ),
    "last_mary": ogmios.Point(
        slot=39916796,
        id="e72579ff89dc9ed325b723a33624b596c08141c7bd573ecfff56a1f7229e4d09",
    ),
    "last_alonzo": ogmios.Point(
        slot=72316796,
        id="c58a24ba8203e7629422a24d9dc68ce2ed495420bf40d9dab124373655161a20",
    ),
    "last_babbage": ogmios.Point(
        slot=133660799,
        id="e757d57eb8dc9500a61c60a39fadb63d9be6973ba96ae337fd24453d4d15c343",
    ),
    "snek_mint": ogmios.Point(
        slot=90914081,
        id="2f7784ab8eee0e3d81223b9bd482195617cbee662ed6c412b123568251aac67a",
    ),
}


# Common Cardano addresses for reference
KNOWN_ADDRESSES = {
    "minswap_v1": "addr1zxn9efv2f6w82hagxqtn62ju4m293tqvw0uhmdl64ch8uw6j2c79gy9l76sdg0xwhd7r0c0kna0tycz4y5s6mlenh8pq6s3z70",
    "sundaeswap_v1": "addr1w9qzpelu9hn45pefc0xr4ac4kdxeswq7pndul2vuj59u8tqaxdznu",
    "wingriders_v1": "addr1w8nvjzjeydcn4atcd93aac8allvrpjn7lx9cyh3rgjgqmrqk5r9ep",
    "snek_mint": "addr1q9jsu6z9sedfksdrhkpmcgvcjf9m6vhd2wn3huxy0s8cwq7k2tc80wsaltznwlfpe7vncdkhcgngll32v22m3g80luvqxjahsf",
}


# Common asset policy IDs for reference
KNOWN_ASSETS = {
    "ada": "",  # Empty string for ADA
    "snek": "279c909f348e533da5808898f87f9a14bb2c3dfbbacccd631d927a3f534e454b",
    "hosky": "a0028f350aaabe0545fdcb56b039bfb08e4bb4d8c4d7c3c7d481c235484f534b59",
    "min": "29d222ce763455e3d7a09a665ce554f00ac89d2e99a1a83d267170c6",
    "sundae": "9a9693a9a37912a5097918f97918d15240c92ab729a0b7c4aa144d77",
}
