import gevent.monkey
gevent.monkey.patch_all()
import socket
import multiprocessing
from gevent import testing as greentest

def block():
    if False:
        i = 10
        return i + 15
    socket.getaddrinfo('localhost', 8001)

class Test(greentest.TestCase):

    def test(self):
        if False:
            return 10
        socket.getaddrinfo('localhost', 8001)
        p = multiprocessing.Process(target=block)
        p.start()
        p.join()
if __name__ == '__main__':
    greentest.main()