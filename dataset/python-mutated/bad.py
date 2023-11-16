import tempfile
files = []

def app(environ, start_response):
    if False:
        for i in range(10):
            print('nop')
    files.append(tempfile.mkstemp())
    start_response('200 OK', [('Content-type', 'text/plain'), ('Content-length', '2')])
    return ['ok']