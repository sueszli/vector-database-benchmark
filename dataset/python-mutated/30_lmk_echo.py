try:
    import network
    import random
    import time
    import espnow
except ImportError:
    print('SKIP')
    raise SystemExit
timeout_ms = 5000
default_pmk = b'MicroPyth0nRules'
default_lmk = b'0123456789abcdef'
sync = True

def echo_server(e):
    if False:
        return 10
    peers = []
    while True:
        (peer, msg) = e.recv(timeout_ms)
        if peer is None:
            return
        if peer not in peers:
            e.add_peer(peer)
        if not e.send(peer, msg, sync):
            print('ERROR: send() failed to', peer)
            return
        if peer not in peers:
            peers.append(peer)
            e.del_peer(peer)
            e.add_peer(peer, default_lmk)
        if msg == b'!done':
            return

def echo_test(e, peer, msg, sync):
    if False:
        for i in range(10):
            print('nop')
    print('TEST: send/recv(msglen=', len(msg), ',sync=', sync, '): ', end='', sep='')
    try:
        if not e.send(peer, msg, sync):
            print('ERROR: Send failed.')
            return
    except OSError as exc:
        print('ERROR: OSError:')
        return
    (p2, msg2) = e.recv(timeout_ms)
    if p2 is None:
        print('ERROR: No response from server.')
        raise SystemExit
    print('OK' if msg2 == msg else 'ERROR: Received != Sent')

def echo_client(e, peer, msglens):
    if False:
        i = 10
        return i + 15
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
        return 10
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
        print('Hello World!')
    e = init(True, False)
    macs = [network.WLAN(i).config('mac') for i in (0, 1)]
    print('Server Start')
    multitest.globals(PEERS=macs)
    multitest.next()
    echo_server(e)
    print('Server Done')
    e.active(False)

def instance1():
    if False:
        for i in range(10):
            print('nop')
    e = init(True, False)
    multitest.next()
    peer = PEERS[0]
    e.add_peer(peer)
    echo_test(e, peer, b'start', True)
    time.sleep(0.1)
    e.del_peer(peer)
    e.add_peer(peer, default_lmk)
    echo_client(e, peer, [250])
    echo_test(e, peer, b'!done', True)
    e.active(False)