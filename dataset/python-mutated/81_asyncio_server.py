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

def client_send(e, peer, msg, sync):
    if False:
        print('Hello World!')
    print('TEST: send/recv(msglen=', len(msg), ',sync=', sync, '): ', end='', sep='')
    try:
        if not e.send(peer, msg, sync):
            print('ERROR: Send failed.')
            return
    except OSError as exc:
        print('ERROR: OSError:')
        return

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

async def echo_server(e):
    peers = []
    async for (peer, msg) in e:
        if peer not in peers:
            peers.append(peer)
            e.add_peer(peer)
        if not await e.asend(peer, msg, sync):
            print('ERROR: asend() failed to', peer)
            return
        if msg == b'!done':
            return

def instance0():
    if False:
        i = 10
        return i + 15
    try:
        import asyncio
        from aioespnow import AIOESPNow
    except ImportError:
        print('SKIP')
        raise SystemExit
    e = AIOESPNow()
    init(e, True, False)
    multitest.globals(PEERS=[network.WLAN(i).config('mac') for i in (0, 1)])
    multitest.next()
    print('Server Start')
    asyncio.run(echo_server(e))
    print('Server Done')
    e.active(False)

def instance1():
    if False:
        for i in range(10):
            print('nop')
    e = espnow.ESPNow()
    init(e, True, False)
    peer = PEERS[0]
    e.add_peer(peer)
    multitest.next()
    for i in range(5):
        msg = bytes([random.getrandbits(8) for _ in range(12)])
        client_send(e, peer, msg, True)
        (p2, msg2) = e.irecv(timeout_ms)
        print('OK' if msg2 == msg else 'ERROR: Received != Sent')
    print('DONE')
    msg = b'!done'
    client_send(e, peer, msg, True)
    (p2, msg2) = e.irecv(timeout_ms)
    print('OK' if msg2 == msg else 'ERROR: Received != Sent')
    e.active(False)