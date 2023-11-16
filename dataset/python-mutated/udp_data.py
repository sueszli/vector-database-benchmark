import socket
NUM_NEW_SOCKETS = 4
NUM_TRANSFERS = 10
TOTAL_PACKETS = NUM_NEW_SOCKETS * NUM_TRANSFERS
PACKET_LOSS_THRESH = 0.75 * TOTAL_PACKETS
PORT = 8000

def print_stats(seq):
    if False:
        while True:
            i = 10
    if TOTAL_PACKETS - seq > PACKET_LOSS_THRESH:
        print('packet loss %.1f%% %d/%d' % ((TOTAL_PACKETS - seq) / TOTAL_PACKETS * 100, seq, TOTAL_PACKETS))
    else:
        print('pass')

def instance0():
    if False:
        while True:
            i = 10
    seq = 0
    multitest.globals(IP=multitest.get_network_ip())
    multitest.next()
    for i in range(NUM_NEW_SOCKETS):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(socket.getaddrinfo('0.0.0.0', PORT + i)[0][-1])
        s.settimeout(0.25)
        multitest.broadcast('server ready')
        for j in range(NUM_TRANSFERS):
            try:
                (data, addr) = s.recvfrom(1000)
            except:
                continue
            if int(data) == seq:
                if seq < TOTAL_PACKETS - PACKET_LOSS_THRESH:
                    print(seq)
                seq += 1
                s.sendto(b'%d' % seq, addr)
        s.close()
    print_stats(seq)

def instance1():
    if False:
        print('Hello World!')
    seq = 0
    multitest.next()
    for i in range(NUM_NEW_SOCKETS):
        ai = socket.getaddrinfo(IP, PORT + i)[0][-1]
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.25)
        multitest.wait('server ready')
        for j in range(NUM_TRANSFERS):
            s.sendto(b'%d' % seq, ai)
            try:
                (data, addr) = s.recvfrom(1000)
            except:
                continue
            if int(data) == seq + 1:
                if seq < TOTAL_PACKETS - PACKET_LOSS_THRESH:
                    print(seq)
                seq += 1
        s.close()
    print_stats(seq)