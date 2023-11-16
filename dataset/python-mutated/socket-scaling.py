import socket
import time
import trio
import trio.testing

async def main():
    for total in [10, 100, 500, 1000, 10000, 20000, 30000]:

        def pt(desc, *, count=total, item='socket'):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal last_time
            now = time.perf_counter()
            total_ms = (now - last_time) * 1000
            per_us = total_ms * 1000 / count
            print(f'{desc}: {total_ms:.2f} ms total, {per_us:.2f} µs/{item}')
            last_time = now
        print(f'\n-- {total} sockets --')
        last_time = time.perf_counter()
        sockets = []
        for _ in range(total // 2):
            (a, b) = socket.socketpair()
            sockets += [a, b]
        pt('socket creation')
        async with trio.open_nursery() as nursery:
            for s in sockets:
                nursery.start_soon(trio.lowlevel.wait_readable, s)
            await trio.testing.wait_all_tasks_blocked()
            pt('spawning wait tasks')
            for _ in range(1000):
                await trio.lowlevel.cancel_shielded_checkpoint()
            pt('scheduling 1000 times', count=1000, item='schedule')
            nursery.cancel_scope.cancel()
        pt('cancelling wait tasks')
        for sock in sockets:
            sock.close()
        pt('closing sockets')
trio.run(main)