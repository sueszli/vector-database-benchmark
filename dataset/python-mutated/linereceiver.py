import time
from twisted.protocols import basic
from twisted.python.compat import range

class CollectingLineReceiver(basic.LineReceiver):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.lines = []
        self.lineReceived = self.lines.append

def deliver(proto, chunks):
    if False:
        while True:
            i = 10
    return [proto.dataReceived(chunk) for chunk in chunks]

def benchmark(chunkSize, lineLength, numLines):
    if False:
        for i in range(10):
            print('nop')
    bytes = (b'x' * lineLength + b'\r\n') * numLines
    chunkCount = len(bytes) // chunkSize + 1
    chunks = []
    for n in range(chunkCount):
        chunks.append(bytes[n * chunkSize:(n + 1) * chunkSize])
    assert b''.join(chunks) == bytes, (chunks, bytes)
    p = CollectingLineReceiver()
    before = time.clock()
    deliver(p, chunks)
    after = time.clock()
    assert bytes.splitlines() == p.lines, (bytes.splitlines(), p.lines)
    print('chunkSize:', chunkSize, end=' ')
    print('lineLength:', lineLength, end=' ')
    print('numLines:', numLines, end=' ')
    print('CPU Time: ', after - before)

def main():
    if False:
        for i in range(10):
            print('nop')
    for numLines in (100, 1000):
        for lineLength in (10, 100, 1000):
            for chunkSize in (1, 500, 5000):
                benchmark(chunkSize, lineLength, numLines)
    for numLines in (10000, 50000):
        for lineLength in (1000, 2000):
            for chunkSize in (51, 500, 5000):
                benchmark(chunkSize, lineLength, numLines)
if __name__ == '__main__':
    main()