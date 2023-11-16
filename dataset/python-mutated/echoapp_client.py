import websocket
from threading import Thread
import time
import sys

def on_message(ws, message):
    if False:
        print('Hello World!')
    print(message)

def on_error(ws, error):
    if False:
        while True:
            i = 10
    print(error)

def on_close(ws, close_status_code, close_msg):
    if False:
        for i in range(10):
            print('nop')
    print('### closed ###')

def on_open(ws):
    if False:
        print('Hello World!')

    def run(*args):
        if False:
            for i in range(10):
                print('nop')
        for i in range(3):
            ws.send('Hello %d' % i)
            time.sleep(1)
        time.sleep(1)
        ws.close()
        print('Thread terminating...')
    Thread(target=run).start()
if __name__ == '__main__':
    websocket.enableTrace(True)
    if len(sys.argv) < 2:
        host = 'ws://echo.websocket.events/'
    else:
        host = sys.argv[1]
    ws = websocket.WebSocketApp(host, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()