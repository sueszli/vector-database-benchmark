import time
import socketio

class Server(socketio.Server):

    def _send_packet(self, eio_sid, pkt):
        if False:
            while True:
                i = 10
        pass

    def _send_eio_packet(self, eio_sid, eio_pkt):
        if False:
            return 10
        pass

def test():
    if False:
        for i in range(10):
            print('nop')
    s = Server()
    start = time.time()
    count = 0
    s._handle_eio_connect('123', 'environ')
    s._handle_eio_message('123', '0')
    while True:
        s.emit('test', 'hello')
        count += 1
        if time.time() - start >= 5:
            break
    return count
if __name__ == '__main__':
    count = test()
    print('server_send:', count, 'packets received.')