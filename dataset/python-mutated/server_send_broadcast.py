import time
import socketio

class Server(socketio.Server):

    def _send_packet(self, eio_sid, pkt):
        if False:
            i = 10
            return i + 15
        pass

    def _send_eio_packet(self, eio_sid, eio_pkt):
        if False:
            print('Hello World!')
        pass

def test():
    if False:
        i = 10
        return i + 15
    s = Server()
    start = time.time()
    count = 0
    for i in range(100):
        s._handle_eio_connect(str(i), 'environ')
        s._handle_eio_message(str(i), '0')
    while True:
        s.emit('test', 'hello')
        count += 1
        if time.time() - start >= 5:
            break
    return count
if __name__ == '__main__':
    count = test()
    print('server_send:', count, 'packets received.')