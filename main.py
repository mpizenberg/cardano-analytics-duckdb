import ogmios
import json


def main():
    print("Hello from fee-analytics!")
    target_addr = "addr1qx0cjs688sse0x7qez9s94aj2eryhd5us4qmha2w5n4kj09uzkt66uw9t5kspx5jwjecx80dz4g33htknafhdhkvzd5szyrk5g"
    snek_mint_addr = "addr1q9jsu6z9sedfksdrhkpmcgvcjf9m6vhd2wn3huxy0s8cwq7k2tc80wsaltznwlfpe7vncdkhcgngll32v22m3g80luvqxjahsf"
    batch_size = 10

    with ogmios.Client() as client:
        print("Connected to host")
        # event = {
        #     "jsonrpc": "2.0",
        #     "method": "queryLedgerState/utxo",
        #     "params": {"addresses": [target_addr]},
        # }
        # client.connection.send(json.dumps(event))
        # print("message sent")
        # response = client.connection.recv()
        # print(f"Received from server: {response}")

        # Set chain pointer to origin
        # point, _, _ = client.find_intersection.execute([ogmios.Origin()])

        # Last babbage block point
        last_babbage_block = ogmios.Point(
            slot=133660799,
            id="e757d57eb8dc9500a61c60a39fadb63d9be6973ba96ae337fd24453d4d15c343",
        )
        # Set chain pointer to the block before the mint of the SNEK token
        block_before_snek_mint = ogmios.Point(
            slot=90914081,
            id="2f7784ab8eee0e3d81223b9bd482195617cbee662ed6c412b123568251aac67a",
        )
        point, tip, _ = client.find_intersection.execute([block_before_snek_mint])

        txs_found = 0
        while True:
            # Batch requests to improve performance
            for i in range(batch_size):
                client.next_block.send()

            for i in range(batch_size):
                direction, tip, block, id = client.next_block.receive()
                if direction.value == "forward":
                    print(f"Block height: {block.height}")
                    # Find transactions involving the target address
                    if isinstance(block, ogmios.Block) and hasattr(
                        block, "transactions"
                    ):
                        for tx in block.transactions:
                            if tx.get("outputs"):
                                for output in tx["outputs"]:
                                    if output["address"] == snek_mint_addr:
                                        txs_found += 1
                                        print(
                                            f"Transaction #{txs_found}: {tx.get('id')}"
                                        )
                                        break

                    # Stop when we've reached the network tip
                    if tip.height == block.height:
                        print(f"Reached chain tip at slot {tip.slot}")
                        return


if __name__ == "__main__":
    main()
