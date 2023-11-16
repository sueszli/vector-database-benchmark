import micropython
try:

    @micropython.bytecode
    def f(x):
        if False:
            while True:
                i = 10
        x and f(x - 1)
    micropython.heap_lock()
    f(1)
    micropython.heap_unlock()
except RuntimeError:
    print('SKIP')
    raise SystemExit
try:
    import asyncio
except ImportError:
    print('SKIP')
    raise SystemExit

class TestStream:

    def __init__(self, blocked):
        if False:
            print('Hello World!')
        self.blocked = blocked

    def write(self, data):
        if False:
            for i in range(10):
                print('nop')
        print('TestStream.write', data)
        if self.blocked:
            return None
        return len(data)

async def task(id, n, t):
    for i in range(n):
        print(id, i)
        await asyncio.sleep_ms(t)

async def main():
    t1 = asyncio.create_task(task(1, 4, 100))
    t2 = asyncio.create_task(task(2, 2, 250))
    micropython.heap_lock()
    print('start')
    await asyncio.sleep_ms(5)
    print('sleep')
    await asyncio.sleep_ms(350)
    print('finish')
    micropython.heap_unlock()
    s = asyncio.StreamWriter(TestStream(True), None)
    micropython.heap_lock()
    s.write(b'12')
    micropython.heap_unlock()
    buf = bytearray(b'56')
    s = asyncio.StreamWriter(TestStream(False), None)
    micropython.heap_lock()
    s.write(b'34')
    s.write(buf)
    micropython.heap_unlock()
asyncio.run(main())