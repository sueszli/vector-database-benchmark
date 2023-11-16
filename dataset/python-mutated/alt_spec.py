def load(arg):
    if False:
        i = 10
        return i + 15

    def app(environ, start_response):
        if False:
            for i in range(10):
                print('nop')
        data = b'Hello, %s!\n' % arg
        status = '200 OK'
        response_headers = [('Content-type', 'text/plain'), ('Content-Length', str(len(data)))]
        start_response(status, response_headers)
        return iter([data])
    return app