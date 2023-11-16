"""Minimal flask webserver serving a Hello World via SSL.

This script gets called as a QProcess from end2end/conftest.py.
"""
import sys
import flask
import webserver_sub
import cheroot.ssl.builtin
app = flask.Flask(__name__)

@app.route('/')
def hello_world():
    if False:
        for i in range(10):
            print('nop')
    return 'Hello World via SSL!'

@app.route('/data/<path:path>')
def send_data(path):
    if False:
        i = 10
        return i + 15
    return webserver_sub.send_data(path)

@app.route('/redirect-http/<path:path>')
def redirect_http(path):
    if False:
        print('Hello World!')
    'Redirect to the given (plaintext) HTTP port on localhost.'
    (host, _orig_port) = flask.request.server
    port = flask.request.args['port']
    return flask.redirect(f'http://{host}:{port}/{path}')

@app.route('/favicon.ico')
def favicon():
    if False:
        for i in range(10):
            print('nop')
    return webserver_sub.favicon()

@app.after_request
def log_request(response):
    if False:
        print('Hello World!')
    return webserver_sub.log_request(response)

def main():
    if False:
        for i in range(10):
            print('nop')
    port = int(sys.argv[1])
    server = webserver_sub.WSGIServer(('127.0.0.1', port), app)
    ssl_dir = webserver_sub.END2END_DIR / 'data' / 'ssl'
    server.ssl_adapter = cheroot.ssl.builtin.BuiltinSSLAdapter(certificate=ssl_dir / 'cert.pem', private_key=ssl_dir / 'key.pem')
    server.start()
if __name__ == '__main__':
    main()