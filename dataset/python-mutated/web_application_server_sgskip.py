"""
=============================================
Embedding in a web application server (Flask)
=============================================

When using Matplotlib in a web server it is strongly recommended to not use
pyplot (pyplot maintains references to the opened figures to make
`~.matplotlib.pyplot.show` work, but this will cause memory leaks unless the
figures are properly closed).

Since Matplotlib 3.1, one can directly create figures using the `.Figure`
constructor and save them to in-memory buffers.  In older versions, it was
necessary to explicitly instantiate an Agg canvas (see e.g.
:doc:`/gallery/user_interfaces/canvasagg`).

The following example uses Flask_, but other frameworks work similarly:

.. _Flask: https://flask.palletsprojects.com

"""
import base64
from io import BytesIO
from flask import Flask
from matplotlib.figure import Figure
app = Flask(__name__)

@app.route('/')
def hello():
    if False:
        print('Hello World!')
    fig = Figure()
    ax = fig.subplots()
    ax.plot([1, 2])
    buf = BytesIO()
    fig.savefig(buf, format='png')
    data = base64.b64encode(buf.getbuffer()).decode('ascii')
    return f"<img src='data:image/png;base64,{data}'/>"