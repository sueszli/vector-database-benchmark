from gunicorn import __version__

def app(environ, start_response):
    if False:
        for i in range(10):
            print('nop')
    'Simplest possible application object'
    status = '200 OK'
    response_headers = [('Content-type', 'text/plain'), ('Transfer-Encoding', 'chunked'), ('X-Gunicorn-Version', __version__)]
    start_response(status, response_headers)
    body = environ['wsgi.input']
    lines = []
    while True:
        line = body.readline()
        if line == b'':
            break
        print(line)
        lines.append(line)
    return iter(lines)