import json
from datetime import datetime
from time import sleep
import numpy as np
from flask import Flask, Response, make_response, request
from bokeh.models import CustomJS, ServerSentDataSource
from bokeh.plotting import figure, show
adapter = CustomJS(code='\n    const result = {x: [], y: []}\n    const pts = cb_data.response\n    for (let i=0; i<pts.length; i++) {\n        result.x.push(pts[i][0])\n        result.y.push(pts[i][1])\n    }\n    return result\n')
source = ServerSentDataSource(data_url='http://localhost:5050/data', max_size=100, mode='append', adapter=adapter)
p = figure(height=800, width=800, background_fill_color='lightgrey', title='Streaming via Server Sent Events', x_range=[-5, 5], y_range=[-5, 5])
p.circle('x', 'y', source=source)
app = Flask(__name__)

def crossdomain(f):
    if False:
        for i in range(10):
            print('nop')

    def wrapped_function(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        resp = make_response(f(*args, **kwargs))
        h = resp.headers
        h['Access-Control-Allow-Origin'] = '*'
        h['Access-Control-Allow-Methods'] = 'GET, OPTIONS, POST'
        h['Access-Control-Max-Age'] = str(21600)
        requested_headers = request.headers.get('Access-Control-Request-Headers')
        if requested_headers:
            h['Access-Control-Allow-Headers'] = requested_headers
        return resp
    return wrapped_function

@app.route('/data', methods=['GET', 'OPTIONS'])
@crossdomain
def stream():
    if False:
        print('Hello World!')

    def event_stream():
        if False:
            i = 10
            return i + 15
        'No global state used'
        while True:
            t = datetime.now().timestamp()
            v = np.sin(t * 5) + 0.2 * np.random.random() + 3
            x = v * np.sin(t)
            y = v * np.cos(t)
            data = [[x, y]]
            yield ('data: ' + json.dumps(data) + '\n\n')
            sleep(0.1)
    resp = Response(event_stream(), mimetype='text/event-stream')
    resp.headers['Cache-Control'] = 'no-cache'
    return resp
show(p)
app.run(port=5050)