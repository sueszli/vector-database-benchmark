import time
from pyinstrument import Profiler
try:
    from flask import Flask, g, make_response, request
except ImportError:
    print('This example requires Flask.')
    print('Install using `pip install flask`.')
    exit(1)
app = Flask(__name__)

@app.before_request
def before_request():
    if False:
        for i in range(10):
            print('nop')
    if 'profile' in request.args:
        g.profiler = Profiler()
        g.profiler.start()

@app.after_request
def after_request(response):
    if False:
        while True:
            i = 10
    if not hasattr(g, 'profiler'):
        return response
    g.profiler.stop()
    output_html = g.profiler.output_html()
    return make_response(output_html)

@app.route('/')
def hello_world():
    if False:
        while True:
            i = 10
    return 'Hello, World!'

@app.route('/sleep')
def sleep():
    if False:
        return 10
    time.sleep(0.1)
    return 'Good morning!'

@app.route('/dosomething')
def do_something():
    if False:
        i = 10
        return i + 15
    import requests
    requests.get('http://google.com')
    return 'Google says hello!'