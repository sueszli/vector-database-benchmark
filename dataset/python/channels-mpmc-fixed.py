import trio
import random


async def main():
    async with trio.open_nursery() as nursery:
        send_channel, receive_channel = trio.open_memory_channel(0)
        async with send_channel, receive_channel:
            # Start two producers, giving each its own private clone
            nursery.start_soon(producer, "A", send_channel.clone())
            nursery.start_soon(producer, "B", send_channel.clone())
            # And two consumers, giving each its own private clone
            nursery.start_soon(consumer, "X", receive_channel.clone())
            nursery.start_soon(consumer, "Y", receive_channel.clone())


async def producer(name, send_channel):
    async with send_channel:
        for i in range(3):
            await send_channel.send(f"{i} from producer {name}")
            # Random sleeps help trigger the problem more reliably
            await trio.sleep(random.random())


async def consumer(name, receive_channel):
    async with receive_channel:
        async for value in receive_channel:
            print(f"consumer {name} got value {value!r}")
            # Random sleeps help trigger the problem more reliably
            await trio.sleep(random.random())


trio.run(main)
