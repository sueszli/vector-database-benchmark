try:
    import time
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
        while True:
            i = 10
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
    if not hasattr(e, 'peers_table'):
        e.active(False)
        print('SKIP')
        raise SystemExit
    e.config(timeout_ms=timeout_ms)
    multitest.next()
    peer = PEERS[0]
    e.add_peer(peer)
    print('IRECV() test...')
    msg = bytes([random.getrandbits(8) for _ in range(12)])
    client_send(e, peer, msg, True)
    (p2, msg2) = e.irecv()
    print('OK' if msg2 == msg else 'ERROR: Received != Sent')
    print('RSSI test...')
    if len(e.peers_table) != 1:
        print('ERROR: len(ESPNow.peers_table()) != 1. ESPNow.peers_table()=', peers)
    elif list(e.peers_table.keys())[0] != peer:
        print('ERROR: ESPNow.peers_table().keys[0] != peer. ESPNow.peers_table()=', peers)
    else:
        (rssi, time_ms) = e.peers_table[peer]
        if not -127 < rssi < 0:
            print('ERROR: Invalid rssi value:', rssi)
        elif time.ticks_diff(time.ticks_ms(), time_ms) > 5000:
            print('ERROR: Unexpected time_ms value:', time_ms)
        else:
            print('OK')
    print('DONE')
    msg = b'!done'
    client_send(e, peer, msg, True)
    (p2, msg2) = e.irecv()
    print('OK' if msg2 == msg else 'ERROR: Received != Sent')
    e.active(False)