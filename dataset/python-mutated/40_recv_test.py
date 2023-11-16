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
        i = 10
        return i + 15
    print('TEST: send/recv(msglen=', len(msg), ',sync=', sync, '): ', end='', sep='')
    try:
        if not e.send(peer, msg, sync):
            print('ERROR: Send failed.')
            return
    except OSError as exc:
        print('ERROR: OSError:')
        return

def init(sta_active=True, ap_active=False):
    if False:
        for i in range(10):
            print('nop')
    wlans = [network.WLAN(i) for i in [network.STA_IF, network.AP_IF]]
    e = espnow.ESPNow()
    e.active(True)
    e.set_pmk(default_pmk)
    wlans[0].active(sta_active)
    wlans[1].active(ap_active)
    wlans[0].disconnect()
    return e

def instance0():
    if False:
        return 10
    e = init(True, False)
    multitest.globals(PEERS=[network.WLAN(i).config('mac') for i in (0, 1)])
    multitest.next()
    print('Server Start')
    echo_server(e)
    print('Server Done')
    e.active(False)

def instance1():
    if False:
        return 10
    e = init(True, False)
    e.config(timeout_ms=timeout_ms)
    multitest.next()
    peer = PEERS[0]
    e.add_peer(peer)
    print('RECVINTO() test...')
    msg = bytes([random.getrandbits(8) for _ in range(12)])
    client_send(e, peer, msg, True)
    data = [bytearray(espnow.ADDR_LEN), bytearray(espnow.MAX_DATA_LEN)]
    n = e.recvinto(data)
    print('OK' if data[1] == msg else 'ERROR: Received != Sent')
    print('IRECV() test...')
    msg = bytes([random.getrandbits(8) for _ in range(12)])
    client_send(e, peer, msg, True)
    (p2, msg2) = e.irecv()
    print('OK' if msg2 == msg else 'ERROR: Received != Sent')
    print('RECV() test...')
    msg = bytes([random.getrandbits(8) for _ in range(12)])
    client_send(e, peer, msg, True)
    (p2, msg2) = e.recv()
    print('OK' if msg2 == msg else 'ERROR: Received != Sent')
    print('ITERATOR() test...')
    msg = bytes([random.getrandbits(8) for _ in range(12)])
    client_send(e, peer, msg, True)
    (p2, msg2) = next(e)
    print('OK' if msg2 == msg else 'ERROR: Received != Sent')
    print('DONE')
    msg = b'!done'
    client_send(e, peer, msg, True)
    (p2, msg2) = e.irecv()
    print('OK' if msg2 == msg else 'ERROR: Received != Sent')
    e.active(False)