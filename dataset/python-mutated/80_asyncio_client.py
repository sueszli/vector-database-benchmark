try:
    import network
    import random
    import espnow
except ImportError:
    print('SKIP')
    raise SystemExit
timeout_ms = 5000
default_pmk = b'MicroPyth0nRules'
sync = True

def echo_server(e):
    if False:
        return 10
    peers = []
    while True:
        (peer, msg) = e.irecv(timeout_ms)
        if peer is None:
            return
        if peer not in peers:
            peers.append(peer)
            e.add_peer(peer)
        if not e.send(peer, msg, sync):
            print('ERROR: send() failed to', peer)
            return
        if msg == b'!done':
            return

def client_send(e, peer, msg, sync):
    if False:
        return 10
    print('TEST: send/recv(msglen=', len(msg), ',sync=', sync, '): ', end='', sep='')
    try:
        if not e.send(peer, msg, sync):
            print('ERROR: Send failed.')
            return
    except OSError as exc:
        print('ERROR: OSError:')
        return
    print('OK')

def init(e, sta_active=True, ap_active=False):
    if False:
        print('Hello World!')
    wlans = [network.WLAN(i) for i in [network.STA_IF, network.AP_IF]]
    e.active(True)
    e.set_pmk(default_pmk)
    wlans[0].active(sta_active)
    wlans[1].active(ap_active)
    wlans[0].disconnect()
    return e

async def client(e):
    init(e, True, False)
    e.config(timeout_ms=timeout_ms)
    peer = PEERS[0]
    e.add_peer(peer)
    multitest.next()
    print('airecv() test...')
    msgs = []
    for i in range(5):
        msgs.append(bytes([random.getrandbits(8) for _ in range(12)]))
        client_send(e, peer, msgs[i], True)
    for i in range(5):
        (mac, reply) = await e.airecv()
        print('OK' if reply == msgs[i] else 'ERROR: Received != Sent')
    print('DONE')
    msg = b'!done'
    client_send(e, peer, msg, True)
    (mac, reply) = await e.airecv()
    print('OK' if reply == msg else 'ERROR: Received != Sent')
    e.active(False)

def instance0():
    if False:
        return 10
    e = espnow.ESPNow()
    init(e, True, False)
    multitest.globals(PEERS=[network.WLAN(i).config('mac') for i in (0, 1)])
    multitest.next()
    print('Server Start')
    echo_server(e)
    print('Server Done')
    e.active(False)

def instance1():
    if False:
        print('Hello World!')
    try:
        import asyncio
        from aioespnow import AIOESPNow
    except ImportError:
        print('SKIP')
        raise SystemExit
    asyncio.run(client(AIOESPNow()))