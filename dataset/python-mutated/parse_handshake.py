"""Benchark parsing WebSocket handshake requests."""
import sys
import timeit
from websockets.http11 import Request
from websockets.streams import StreamReader
CHROME_HANDSHAKE = b'GET / HTTP/1.1\r\nHost: localhost:5678\r\nConnection: Upgrade\r\nPragma: no-cache\r\nCache-Control: no-cache\r\nUser-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36\r\nUpgrade: websocket\r\nOrigin: null\r\nSec-WebSocket-Version: 13\r\nAccept-Encoding: gzip, deflate, br\r\nAccept-Language: en-GB,en;q=0.9,en-US;q=0.8,fr;q=0.7\r\nSec-WebSocket-Key: ebkySAl+8+e6l5pRKTMkyQ==\r\nSec-WebSocket-Extensions: permessage-deflate; client_max_window_bits\r\n\r\n'
FIREFOX_HANDSHAKE = b'GET / HTTP/1.1\r\nHost: localhost:5678\r\nUser-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/111.0\r\nAccept: */*\r\nAccept-Language: en-US,en;q=0.7,fr-FR;q=0.3\r\nAccept-Encoding: gzip, deflate, br\r\nSec-WebSocket-Version: 13\r\nOrigin: null\r\nSec-WebSocket-Extensions: permessage-deflate\r\nSec-WebSocket-Key: 1PuS+hnb+0AXsL7z2hNAhw==\r\nConnection: keep-alive, Upgrade\r\nSec-Fetch-Dest: websocket\r\nSec-Fetch-Mode: websocket\r\nSec-Fetch-Site: cross-site\r\nPragma: no-cache\r\nCache-Control: no-cache\r\nUpgrade: websocket\r\n\r\n'
WEBSOCKETS_HANDSHAKE = b'GET / HTTP/1.1\r\nHost: localhost:8765\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: 9c55e0/siQ6tJPCs/QR8ZA==\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Extensions: permessage-deflate; client_max_window_bits\r\nUser-Agent: Python/3.11 websockets/11.0\r\n\r\n'

def parse_handshake(handshake):
    if False:
        for i in range(10):
            print('nop')
    reader = StreamReader()
    reader.feed_data(handshake)
    parser = Request.parse(reader.read_line)
    try:
        next(parser)
    except StopIteration:
        pass
    else:
        assert False, 'parser should return request'
    reader.feed_eof()
    assert reader.at_eof(), 'parser should consume all data'

def run_benchmark(name, handshake, number=10000):
    if False:
        i = 10
        return i + 15
    ph = min(timeit.repeat('parse_handshake(handshake)', number=number, globals={'parse_handshake': parse_handshake, 'handshake': handshake})) / number * 1000000
    print(f'{name}\t{len(handshake)}\t{ph:.1f}')
if __name__ == '__main__':
    print('Sizes are in bytes. Times are in µs per frame.', file=sys.stderr)
    print('Run `tabs -16` for clean output. Pipe stdout to TSV for saving.')
    print(file=sys.stderr)
    print('client\tsize\ttime')
    run_benchmark('Chrome', CHROME_HANDSHAKE)
    run_benchmark('Firefox', FIREFOX_HANDSHAKE)
    run_benchmark('websockets', WEBSOCKETS_HANDSHAKE)