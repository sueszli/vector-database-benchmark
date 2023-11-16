from aiohttp import web
from aiohttp_wsgi import WSGIHandler
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    if False:
        return 10
    return 'Hello, world!'

def make_aiohttp_app(app):
    if False:
        for i in range(10):
            print('nop')
    wsgi_handler = WSGIHandler(app)
    aioapp = web.Application()
    aioapp.router.add_route('*', '/{path_info:.*}', wsgi_handler)
    return aioapp
aioapp = make_aiohttp_app(app)