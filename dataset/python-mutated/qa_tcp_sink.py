from gnuradio import gr, gr_unittest, blocks, network
import socket
import threading
import time

class qa_tcp_sink(gr_unittest.TestCase):

    def tcp_receive(self, serversocket):
        if False:
            print('Hello World!')
        for _ in range(2):
            (clientsocket, address) = serversocket.accept()
            while True:
                data = clientsocket.recv(4096)
                if not data:
                    break
            clientsocket.close()

    def setUp(self):
        if False:
            while True:
                i = 10
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = None

    def test_restart(self):
        if False:
            while True:
                i = 10
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serversocket.bind(('localhost', 2000))
        serversocket.listen()
        thread = threading.Thread(target=self.tcp_receive, args=(serversocket,))
        thread.start()
        null_source = blocks.null_source(gr.sizeof_gr_complex)
        throttle = blocks.throttle(gr.sizeof_gr_complex, 320000, True)
        tcp_sink = network.tcp_sink(gr.sizeof_gr_complex, 1, '127.0.0.1', 2000, 1)
        self.tb.connect(null_source, throttle, tcp_sink)
        self.tb.start()
        time.sleep(0.1)
        self.tb.stop()
        time.sleep(0.1)
        self.tb.start()
        time.sleep(0.1)
        self.tb.stop()
        thread.join()
        serversocket.close()
if __name__ == '__main__':
    gr_unittest.run(qa_tcp_sink)