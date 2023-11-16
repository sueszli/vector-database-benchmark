import websocket
import json
import traceback
from threading import Thread, Condition
from base64 import b64encode

class MockSteam:

    def __init__(self, base_uri, username, password, trace=False):
        if False:
            print('Hello World!')
        uri = 'ws://%s/3/Steam.websocket' % base_uri
        if username is not None:
            userAndPass = b64encode(('%s:%s' % (username, password)).encode('ascii')).decode('ascii')
            headers = ['Authorization: Basic %s' % userAndPass]
        else:
            headers = None
        self.messages = []
        self.connected = False
        self.connected_lock = Condition()
        self.have_message = Condition()

        def on_msg(ws, msg):
            if False:
                for i in range(10):
                    print('nop')
            self.on_message(ws, msg)

        def on_opn(ws):
            if False:
                for i in range(10):
                    print('nop')
            self.on_open(ws)
        if trace:
            websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp(uri, header=headers, on_message=on_msg, on_open=on_opn)

        def run(*args):
            if False:
                while True:
                    i = 10
            print('run(%s, %s)' % (uri, headers))
            try:
                self.ws.run_forever()
            except:
                traceback.print_exc()
            finally:
                print('run(-)')
        client = Thread(target=run, daemon=True)
        client.start()
        try:
            self.connected_lock.acquire()
            self.connected_lock.wait(10)
            if not self.connected:
                raise Exception('Timeout connecting to H2O.')
            else:
                print('Connected')
        finally:
            self.connected_lock.release()

    def close(self):
        if False:
            i = 10
            return i + 15
        self.ws.close()

    def on_message(self, ws, message):
        if False:
            while True:
                i = 10
        print('Received: %s' % message)
        try:
            self.have_message.acquire()
            self.messages.append(json.loads(message))
            self.have_message.notify_all()
        finally:
            self.have_message.release()

    def on_open(self, ws):
        if False:
            while True:
                i = 10
        try:
            self.connected_lock.acquire()
            self.connected = True
            self.connected_lock.notify_all()
        finally:
            self.connected_lock.release()

    def wait_for_message(self, timeout=5):
        if False:
            while True:
                i = 10
        try:
            self.have_message.acquire()
            if len(self.messages) == 0:
                self.have_message.wait(timeout)
            if len(self.messages) > 0:
                return self.messages.pop()
            else:
                return None
        finally:
            self.have_message.release()

    def send(self, msg):
        if False:
            i = 10
            return i + 15
        self.ws.send(json.dumps(msg))
        print('Sent: %s' % msg)