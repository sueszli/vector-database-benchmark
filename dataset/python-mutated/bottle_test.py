import bottle
app = bottle.Bottle()

@app.route('/text')
def text():
    if False:
        print('Hello World!')
    return 'Hello, world!'