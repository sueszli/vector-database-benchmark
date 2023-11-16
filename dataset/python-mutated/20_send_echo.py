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
        i = 10
        return i + 15
    peers = []
    while True:
        (peer, msg) = e.recv(timeout_ms)
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

def echo_test(e, peer, msg, sync):
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
    (p2, msg2) = e.recv(timeout_ms)
    print('OK' if msg2 == msg else 'ERROR: Received != Sent')

def echo_client(e, peer, msglens):
    if False:
        while True:
            i = 10
    for sync in [True, False]:
        for msglen in msglens:
            msg = bytearray(msglen)
            if msglen > 0:
                msg[0] = b'_'[0]
            for i in range(1, msglen):
                msg[i] = random.getrandbits(8)
            echo_test(e, peer, msg, sync)

def init(sta_active=True, ap_active=False):
    if False:
        while True:
            i = 10
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
    multitest.next()
    peer = PEERS[0]
    e.add_peer(peer)
    echo_client(e, peer, [1, 2, 8, 100, 249, 250, 251, 0])
    echo_test(e, peer, b'!done', True)
    e.active(False)