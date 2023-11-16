from sanic import Sanic
from sanic.response import html
import socketio
sio = socketio.AsyncServer(async_mode='sanic')
app = Sanic()
sio.attach(app)

@app.route('/')
def index(request):
    if False:
        return 10
    with open('fiddle.html') as f:
        return html(f.read())

@sio.event
async def connect(sid, environ, auth):
    print(f'connected auth={auth} sid={sid}')
    await sio.emit('hello', (1, 2, {'hello': 'you'}), to=sid)

@sio.event
def disconnect(sid):
    if False:
        for i in range(10):
            print('nop')
    print('disconnected', sid)
app.static('/static', './static')
if __name__ == '__main__':
    app.run()