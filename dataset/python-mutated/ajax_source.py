import numpy as np
from flask import Flask, jsonify, make_response, request
from bokeh.models import AjaxDataSource, CustomJS
from bokeh.plotting import figure, show
adapter = CustomJS(code='\n    const result = {x: [], y: []}\n    const {points} = cb_data.response\n    for (const [x, y] of points) {\n        result.x.push(x)\n        result.y.push(y)\n    }\n    return result\n')
source = AjaxDataSource(data_url='http://localhost:5050/data', polling_interval=100, adapter=adapter)
p = figure(height=300, width=800, background_fill_color='lightgrey', title='Streaming Noisy sin(x) via Ajax')
p.circle('x', 'y', source=source)
p.x_range.follow = 'end'
p.x_range.follow_interval = 10
app = Flask(__name__)

def crossdomain(f):
    if False:
        for i in range(10):
            print('nop')

    def wrapped_function(*args, **kwargs):
        if False:
            print('Hello World!')
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
x = list(np.arange(0, 6, 0.1))
y = list(np.sin(x) + np.random.random(len(x)))

@app.route('/data', methods=['GET', 'OPTIONS', 'POST'])
@crossdomain
def data():
    if False:
        for i in range(10):
            print('nop')
    x.append(x[-1] + 0.1)
    y.append(np.sin(x[-1]) + np.random.random())
    return jsonify(points=list(zip(x, y)))
show(p)
app.run(port=5050)