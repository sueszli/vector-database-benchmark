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
        print('Hello World!')
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
        i = 10
        return i + 15
    e = init(True, False)
    multitest.globals(PEERS=[network.WLAN(i).config('mac') for i in (0, 1)])
    multitest.next()
    print('Server Start')
    echo_server(e)
    print('Server Done')
    e.active(False)
done = False

def instance1():
    if False:
        while True:
            i = 10
    e = init(True, False)
    try:
        e.irq(None)
    except AttributeError:
        print('SKIP')
        raise SystemExit
    e.config(timeout_ms=timeout_ms)
    multitest.next()
    peer = PEERS[0]
    e.add_peer(peer)

    def on_recv_cb(e):
        if False:
            i = 10
            return i + 15
        global done
        (p2, msg2) = e.irecv(0)
        print('OK' if msg2 == msg else 'ERROR: Received != Sent')
        done = True
    global done
    print('IRQ() test...')
    e.irq(on_recv_cb)
    done = False
    msg = bytes([random.getrandbits(8) for _ in range(12)])
    client_send(e, peer, msg, True)
    start = time.ticks_ms()
    while not done:
        if time.ticks_diff(time.ticks_ms(), start) > timeout_ms:
            print('Timeout waiting for response.')
            raise SystemExit
    e.irq(None)
    print('DONE')
    msg = b'!done'
    client_send(e, peer, msg, True)
    (p2, msg2) = e.irecv()
    print('OK' if msg2 == msg else 'ERROR: Received != Sent')
    e.active(False)