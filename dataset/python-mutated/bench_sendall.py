from __future__ import print_function, division, absolute_import
import perf
from gevent import socket
from gevent.server import StreamServer

def recvall(sock, _):
    if False:
        while True:
            i = 10
    while sock.recv(4096):
        pass
N = 10
runs = []

def benchmark(conn, data):
    if False:
        return 10
    spent_total = 0
    for _ in range(N):
        start = perf.perf_counter()
        conn.sendall(data)
        spent = perf.perf_counter() - start
        spent_total += spent
    runs.append(spent_total)
    return spent_total

def main():
    if False:
        while True:
            i = 10
    runner = perf.Runner()
    server = StreamServer(('127.0.0.1', 0), recvall)
    server.start()
    MB = 1024 * 1024
    length = 50 * MB
    data = b'x' * length
    conn = socket.create_connection((server.server_host, server.server_port))
    runner.bench_func('sendall', benchmark, conn, data, inner_loops=N)
    conn.close()
    server.stop()
    if runs:
        total = sum(runs)
        avg = total / len(runs)
        print('~ %.2f MB/s' % (length * N / avg / MB))
if __name__ == '__main__':
    main()