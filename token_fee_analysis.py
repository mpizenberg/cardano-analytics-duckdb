import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import argparse


class TokenFeeAnalyzer:
    """
    Analyzer for querying token transactions and aggregating fees using DuckDB.

    This class provides methods to identify transactions involving specific tokens
    where the token changes ownership (transfers), and aggregate the fees for
    these transactions.

    PRECISE OWNERSHIP DETECTION:
    The parquet schema includes input UTXO references, allowing precise implementation
    of the ownership change rules:
    1. Token appears in outputs OR in a mint
    2. When there is no mint, at least one address must be different between
       UTXOs holding the token in inputs vs outputs

    Transactions referencing input UTXOs not present in the database (older than
    the data range) are filtered out and reported separately.
    """

    def __init__(self, data_dir: str):
        """
        Initialize the TokenFeeAnalyzer.

        Args:
            data_dir: Path to the directory containing parquet files
        """
        self.data_dir = Path(data_dir)
        self.conn = duckdb.connect()
        self._setup_views()

    def _setup_views(self):
        """Set up DuckDB views for the parquet files."""
        print("Setting up DuckDB views...")

        # Create views for each parquet file type across all slot groups
        slot_dirs = [
            d
            for d in self.data_dir.iterdir()
            if d.is_dir() and d.name.startswith("slot_")
        ]

        if not slot_dirs:
            raise ValueError(f"No slot directories found in {self.data_dir}")

        # Build file patterns for each data type
        file_patterns = {"tx": [], "asset": [], "utxo": [], "mint": []}

        for slot_dir in slot_dirs:
            for file_type in file_patterns.keys():
                parquet_file = slot_dir / f"{file_type}.parquet"
                if parquet_file.exists():
                    file_patterns[file_type].append(str(parquet_file))

        # Create views if files exist
        for file_type, files in file_patterns.items():
            if files:
                file_list = "', '".join(files)
                query = f"""
                CREATE OR REPLACE VIEW {file_type}_view AS
                SELECT * FROM read_parquet(['{file_list}'])
                """
                self.conn.execute(query)
                print(f"Created {file_type}_view with {len(files)} files")

    def get_token_info(self, token_name: str) -> Optional[Dict[str, str]]:
        """
        Get token information for well-known tokens.

        Args:
            token_name: Name of the token (e.g., 'snek')

        Returns:
            Dictionary with policy_id and asset_name, or None if not found
        """
        # Well-known token mappings
        known_tokens = {
            "snek": {
                "policy_id": "279c909f348e533da5808898f87f9a14bb2c3dfbbacccd631d927a3f",
                "asset_name": "534e454b",  # 'SNEK' in hex
            },
            "hosky": {
                "policy_id": "a0028f350aaabe0545fdcb56b039bfb08e4bb4d8c4d7c3c7d481c235",
                "asset_name": "484f534b59",  # 'HOSKY' in hex
            },
        }

        return known_tokens.get(token_name.lower())

    def find_token_utxos(
        self,
        policy_id: str,
        asset_name: str,
        min_slot: Optional[int] = None,
        max_slot: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Find all UTXOs containing a specific token for debugging purposes.

        Args:
            policy_id: The policy ID of the token (hex string)
            asset_name: The asset name of the token (hex string)
            min_slot: Optional minimum slot number to filter by
            max_slot: Optional maximum slot number to filter by

        Returns:
            DataFrame with all UTXOs containing the token
        """
        print(f"Searching for UTXOs containing token...")
        print(f"Policy ID: {policy_id}")
        print(f"Asset Name: {asset_name}")

        # Convert hex strings to binary format for querying
        # Use unhex() function to properly convert hex strings to binary data
        policy_id_binary = f"unhex('{policy_id}')"
        asset_name_binary = f"unhex('{asset_name}')" if asset_name else "NULL"

        # Base query to find all UTXOs with the token
        slot_filter = ""
        if min_slot is not None or max_slot is not None:
            conditions = []
            if min_slot is not None:
                conditions.append(f"a.slot >= {min_slot}")
            if max_slot is not None:
                conditions.append(f"a.slot <= {max_slot}")
            slot_filter = f"AND {' AND '.join(conditions)}"

        query = f"""
        SELECT
            a.slot,
            a.tx_id,
            a.output_index,
            a.policy_id,
            a.asset_name,
            a.amount,
            u.address,
            u.lovelace
        FROM asset_view a
        JOIN utxo_view u ON (
            a.tx_id = u.tx_id AND
            a.output_index = u.output_index AND
            a.slot = u.slot
        )
        WHERE a.policy_id = {policy_id_binary}
            AND a.asset_name = {asset_name_binary}
            {slot_filter}
        ORDER BY a.slot ASC
        """

        print("Executing UTXO query...")
        result = self.conn.execute(query).fetchdf()
        print(f"Found {len(result)} UTXOs containing the token")

        return result

    def find_token_transfers(
        self,
        policy_id: str,
        asset_name: str,
        min_slot: Optional[int] = None,
        max_slot: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Find transactions involving a specific token where ownership changed.

        This method works in steps:
        1. Find all UTXOs containing the token (token_utxos)
        2. Find transactions that involve these UTXOs (either as inputs or outputs)
        3. Apply ownership change rules based on input vs output address sets

        Args:
            policy_id: The policy ID of the token (hex string)
            asset_name: The asset name of the token (hex string)
            min_slot: Optional minimum slot number to filter by
            max_slot: Optional maximum slot number to filter by

        Returns:
            DataFrame with transaction details including fees
        """
        print(f"Step-by-step token transfer analysis...")

        # Step 1: Find all UTXOs containing the token
        token_utxos = self.find_token_utxos(policy_id, asset_name, min_slot, max_slot)

        if len(token_utxos) == 0:
            print("No UTXOs found containing this token!")
            return pd.DataFrame()

        print(f"Step 1 complete: Found {len(token_utxos)} UTXOs containing the token")

        # Step 2: Find all transactions that either create or consume these UTXOs
        print("Step 2: Finding transactions that involve these UTXOs...")

        # Convert hex strings to binary format for querying
        # Use unhex() function to properly convert hex strings to binary data
        policy_id_binary = f"unhex('{policy_id}')"
        asset_name_binary = f"unhex('{asset_name}')"

        # Base slot filter
        slot_filter = ""
        if min_slot is not None or max_slot is not None:
            conditions = []
            if min_slot is not None:
                conditions.append(f"t.slot >= {min_slot}")
            if max_slot is not None:
                conditions.append(f"t.slot <= {max_slot}")
            slot_filter = f"AND {' AND '.join(conditions)}"

        query = f"""
        WITH relevant_txs AS (
            -- All transactions with the token in outputs
            SELECT DISTINCT a.tx_id, t.slot, t.tx_fee
            FROM asset_view a
            JOIN tx_view t ON a.tx_id = t.tx_id
            WHERE a.policy_id = {policy_id_binary}
              AND a.asset_name = {asset_name_binary}
              {slot_filter if slot_filter else ""}
        ),

        input_addresses AS (
            -- Addresses of input UTxOs holding that token
            SELECT
                tx.tx_id,
                ARRAY_SORT(ARRAY_AGG(DISTINCT u_in.address)) AS input_addr_set
            FROM tx_view tx,
                 UNNEST(tx.inputs) AS t(input_ref)
            JOIN asset_view a_in
              ON a_in.tx_id = input_ref.tx_id
             AND a_in.output_index = input_ref.output_index
             AND a_in.policy_id = {policy_id_binary}
             AND a_in.asset_name = {asset_name_binary}
            JOIN utxo_view u_in
              ON u_in.tx_id = a_in.tx_id
             AND u_in.output_index = a_in.output_index
            GROUP BY tx.tx_id
        ),

        output_addresses AS (
            -- Addresses of output UTxOs holding that token
            SELECT
                a_out.tx_id,
                ARRAY_SORT(ARRAY_AGG(DISTINCT u_out.address)) AS output_addr_set
            FROM asset_view a_out
            JOIN utxo_view u_out
              ON u_out.tx_id = a_out.tx_id
             AND u_out.output_index = a_out.output_index
            WHERE a_out.policy_id = {policy_id_binary}
              AND a_out.asset_name = {asset_name_binary}
            GROUP BY a_out.tx_id
        )

        SELECT
            tx.slot,
            tx.tx_id,
            tx.tx_fee,
            input_addresses.input_addr_set,
            output_addresses.output_addr_set
        FROM relevant_txs tx
        LEFT JOIN input_addresses USING (tx_id)
        LEFT JOIN output_addresses USING (tx_id)
        WHERE input_addresses.input_addr_set <> output_addresses.output_addr_set
        ORDER BY tx.slot ASC;
        """

        print("Executing preliminary transaction query...")
        result = self.conn.execute(query).fetchdf()
        print(f"Found {len(result)} transactions that involve token UTXOs")

        if len(result) == 0:
            print("No transactions found that involve these UTXOs!")
            return pd.DataFrame()

        return result

    def analyze_token_fees(
        self,
        token_name: str,
        min_slot: Optional[int] = None,
        max_slot: Optional[int] = None,
        save_details: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze fees for transactions involving a specific token.

        Args:
            token_name: Name of the token (e.g., 'snek')
            min_slot: Optional minimum slot number
            max_slot: Optional maximum slot number
            save_details: Whether to save detailed results to CSV

        Returns:
            Dictionary with analysis results
        """
        # Get token information
        token_info = self.get_token_info(token_name)
        if not token_info:
            raise ValueError(
                f"Unknown token: {token_name}. Please provide policy_id and asset_name manually."
            )

        print(f"Analyzing fees for {token_name} token...")

        # Find token transfers (preliminary)
        transfers = self.find_token_transfers(
            token_info["policy_id"],
            token_info["asset_name"],
            min_slot=min_slot,
            max_slot=max_slot,
        )

        if len(transfers) == 0:
            print(f"No transactions found involving {token_name} token!")
            return {
                "token_name": token_name,
                "total_transactions": 0,
                "total_fees_lovelace": 0,
                "total_fees_ada": 0,
                "avg_fee_lovelace": 0,
                "avg_fee_ada": 0,
            }

        # Calculate statistics
        total_transactions = len(transfers)
        total_fees_lovelace = int(transfers["tx_fee"].sum())
        total_fees_ada = total_fees_lovelace / 1_000_000
        avg_fee_lovelace = int(transfers["tx_fee"].mean())
        avg_fee_ada = avg_fee_lovelace / 1_000_000

        # Additional statistics by transfer type if available
        transfer_type_stats = {}
        if "transfer_type" in transfers.columns:
            transfer_type_stats = (
                transfers.groupby("transfer_type")
                .agg({"tx_fee": ["count", "sum", "mean"]})
                .round(2)
                .to_dict()
            )

        # Slot range analysis
        min_slot_found = int(transfers["slot"].min())
        max_slot_found = int(transfers["slot"].max())

        results = {
            "token_name": token_name,
            "policy_id": token_info["policy_id"],
            "asset_name": token_info["asset_name"],
            "total_transactions": total_transactions,
            "total_fees_lovelace": total_fees_lovelace,
            "total_fees_ada": total_fees_ada,
            "avg_fee_lovelace": avg_fee_lovelace,
            "avg_fee_ada": avg_fee_ada,
            "slot_range": (min_slot_found, max_slot_found),
            "transfer_type_breakdown": transfer_type_stats,
        }

        # Save detailed results
        if save_details:
            output_file = (
                f"{token_name}_transfers_{min_slot_found}_{max_slot_found}.csv"
            )
            transfers.to_csv(output_file, index=False)
            print(f"Detailed results saved to {output_file}")

        return results

    def print_analysis_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of the analysis results."""
        print(f"\n{'=' * 60}")
        print(f"TOKEN FEE ANALYSIS SUMMARY - {results['token_name'].upper()}")
        print(f"{'=' * 60}")
        print(f"Policy ID: {results['policy_id']}")
        print(f"Asset Name: {results['asset_name']}")
        print(
            f"Slot Range: {results['slot_range'][0]:,} - {results['slot_range'][1]:,}"
        )
        print(f"\nTRANSACTION METRICS:")
        print(f"  Total Transactions: {results['total_transactions']:,}")
        print(
            f"  Total Fees: {results['total_fees_lovelace']:,} lovelace ({results['total_fees_ada']:.6f} ADA)"
        )
        print(
            f"  Average Fee: {results['avg_fee_lovelace']:,} lovelace ({results['avg_fee_ada']:.6f} ADA)"
        )

        if "transfer_type_breakdown" in results:
            print(f"\nTRANSFER TYPE BREAKDOWN:")
            for transfer_type, stats in results["transfer_type_breakdown"].items():
                if isinstance(stats, dict) and "tx_fee" in stats:
                    count = stats["tx_fee"]["count"]
                    total_fee = stats["tx_fee"]["sum"]
                    avg_fee = stats["tx_fee"]["mean"]
                    print(
                        f"  {transfer_type}: {count} txs, {total_fee:,.0f} total fee, {avg_fee:,.0f} avg fee"
                    )

        print(
            f"\nNOTE: Analysis uses precise ownership change detection with input UTXO data."
        )
        print(
            f"Transactions filtered out due to missing input UTXOs are reported separately."
        )

    def test_token_analysis(self, token_name: str) -> Dict[str, Any]:
        """
        Simple test method to debug token analysis step by step.

        Args:
            token_name: Name of the token to test (e.g., 'snek')

        Returns:
            Dictionary with debug information
        """
        print(f"\n{'=' * 50}")
        print(f"DEBUGGING TOKEN ANALYSIS: {token_name.upper()}")
        print(f"{'=' * 50}")

        # Get token info
        token_info = self.get_token_info(token_name)
        if not token_info:
            print(f"ERROR: Unknown token {token_name}")
            return {"error": f"Unknown token {token_name}"}

        print(f"Policy ID: {token_info['policy_id']}")
        print(f"Asset Name: {token_info['asset_name']}")

        # Step 1: Test basic UTXO finding
        print(f"\nStep 1: Finding UTXOs with this token...")
        token_utxos = self.find_token_utxos(
            token_info["policy_id"], token_info["asset_name"]
        )

        debug_info = {
            "token_name": token_name,
            "policy_id": token_info["policy_id"],
            "asset_name": token_info["asset_name"],
            "utxos_found": len(token_utxos),
        }

        if len(token_utxos) > 0:
            # Show some stats about the UTXOs
            unique_addresses = token_utxos["address"].nunique()
            unique_txs = token_utxos["tx_id"].nunique()
            slot_range = (token_utxos["slot"].min(), token_utxos["slot"].max())
            total_amount = token_utxos["amount"].sum()

            print(f"  UTXOs found: {len(token_utxos)}")
            print(f"  Unique addresses: {unique_addresses}")
            print(f"  Unique transactions: {unique_txs}")
            print(f"  Slot range: {slot_range[0]:,} - {slot_range[1]:,}")
            print(f"  Total token amount: {total_amount:,}")

            debug_info.update(
                {
                    "unique_addresses": unique_addresses,
                    "unique_transactions": unique_txs,
                    "slot_range": slot_range,
                    "total_token_amount": total_amount,
                }
            )

            # Show first few UTXOs as sample
            print(f"\nFirst 5 UTXOs:")
            sample_utxos = token_utxos.head()
            for _, utxo in sample_utxos.iterrows():
                print(
                    f"  Slot {utxo['slot']:,}: {utxo['amount']:,} tokens at {utxo['address'][:20]}..."
                )

        print(f"\nDEBUG COMPLETE")
        print(f"{'=' * 50}")

        return debug_info

    def close(self):
        """Close the DuckDB connection."""
        self.conn.close()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Analyze fees for token transactions")
    parser.add_argument("data_dir", help="Path to directory containing parquet files")
    parser.add_argument("token_name", help="Name of the token to analyze (e.g., snek)")
    parser.add_argument("--min-slot", type=int, help="Minimum slot number to analyze")
    parser.add_argument("--max-slot", type=int, help="Maximum slot number to analyze")
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save detailed results to CSV"
    )

    args = parser.parse_args()

    try:
        # Initialize analyzer
        analyzer = TokenFeeAnalyzer(args.data_dir)

        # Perform analysis
        results = analyzer.analyze_token_fees(
            args.token_name,
            min_slot=args.min_slot,
            max_slot=args.max_slot,
            save_details=not args.no_save,
        )

        # Print summary
        analyzer.print_analysis_summary(results)

        # Close connection
        analyzer.close()

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
