import trio

def thread_fn(receive_from_trio, send_to_trio):
    if False:
        print('Hello World!')
    while True:
        try:
            request = trio.from_thread.run(receive_from_trio.receive)
        except trio.EndOfChannel:
            trio.from_thread.run(send_to_trio.aclose)
            return
        else:
            response = request + 1
            trio.from_thread.run(send_to_trio.send, response)

async def main():
    (send_to_thread, receive_from_trio) = trio.open_memory_channel(0)
    (send_to_trio, receive_from_thread) = trio.open_memory_channel(0)
    async with trio.open_nursery() as nursery:
        nursery.start_soon(trio.to_thread.run_sync, thread_fn, receive_from_trio, send_to_trio)
        await send_to_thread.send(0)
        print(await receive_from_thread.receive())
        await send_to_thread.send(1)
        print(await receive_from_thread.receive())
        await send_to_thread.aclose()
trio.run(main)