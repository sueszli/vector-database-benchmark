try:
    import asyncio
except ImportError:
    print('SKIP')
    raise SystemExit
PORT = 8000

async def handle_connection(reader, writer):
    writer.write(b'x')
    await writer.drain()
    print('read:', await reader.read(100))
    print('close')
    writer.close()
    await writer.wait_closed()
    print('done')
    ev.set()

async def tcp_server():
    global ev
    ev = asyncio.Event()
    server = await asyncio.start_server(handle_connection, '0.0.0.0', PORT)
    print('server running')
    multitest.next()
    async with server:
        await asyncio.wait_for(ev.wait(), 10)

async def tcp_client():
    (reader, writer) = await asyncio.open_connection(IP, PORT)
    print('read:', await reader.read(1))
    print('close')
    writer.close()
    await writer.wait_closed()
    print('write')
    try:
        writer.write(b'x')
        await writer.drain()
    except OSError:
        print('OSError')

def instance0():
    if False:
        print('Hello World!')
    multitest.globals(IP=multitest.get_network_ip())
    asyncio.run(tcp_server())

def instance1():
    if False:
        return 10
    multitest.next()
    asyncio.run(tcp_client())