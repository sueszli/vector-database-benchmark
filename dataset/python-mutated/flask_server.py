"""This example demonstrates embedding an autoloaded Bokeh server
into a simple Flask application, and passing arguments to Bokeh.

To view the example, run:

    python flask_server.py

in this directory, and navigate to:

    http://localhost:5000

"""
import atexit
import subprocess
from flask import Flask, render_template_string
from bokeh.embed import server_document
home_html = '\n<!DOCTYPE html>\n<html lang="en">\n  <body>\n    <h1><a href="/batch/1"> Batch 1 (cos)</a></h1>\n    <h1><a href="/batch/2"> Batch 2 (sin)</a></h1>\n    <h1><a href="/batch/3"> Batch 3 (tan)</a></h1>\n  </body>\n</html>\n'
app_html = '\n<!DOCTYPE html>\n<html lang="en">\n  <body>\n    <div>\n        <h2><a href="/batch/1">Batch 1 (cos)</a> - <a href="/batch/2">Batch 2 (sin)</a> - <a href="/batch/3">Batch 3 (tan)</a></h2>\n    </div>\n    {{ bokeh_script|safe }}\n  </body>\n</html>\n'
app = Flask(__name__)
bokeh_process = subprocess.Popen(['python', '-m', 'bokeh', 'serve', '--allow-websocket-origin=localhost:5000', 'bokeh_server.py'], stdout=subprocess.PIPE)

@atexit.register
def kill_server():
    if False:
        return 10
    bokeh_process.kill()

@app.route('/')
def home():
    if False:
        while True:
            i = 10
    return render_template_string(home_html)

@app.route('/batch/<int:batchid>')
def visualization(batchid):
    if False:
        while True:
            i = 10
    bokeh_script = server_document(url='http://localhost:5006/bokeh_server', arguments=dict(batchid=batchid))
    return render_template_string(app_html, bokeh_script=bokeh_script)
if __name__ == '__main__':
    app.run(debug=True)