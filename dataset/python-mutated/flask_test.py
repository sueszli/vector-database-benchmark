import flask
app = flask.Flask(__name__)

@app.route('/text')
def text():
    if False:
        for i in range(10):
            print('nop')
    return 'Hello, world!'