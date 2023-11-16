@anno1
def bar(foo: 'foo'):
    if False:
        return 10
    x = 2

@anno1
@anno2
def foobar():
    if False:
        print('Hello World!')
    x = 3

@app.route('/foo')
def foo():
    if False:
        print('Hello World!')
    x = 1

def no_anno():
    if False:
        return 10
    x = 2