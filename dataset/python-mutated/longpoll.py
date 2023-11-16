import sys
import time

class TestIter(object):

    def __iter__(self):
        if False:
            while True:
                i = 10
        lines = [b'line 1\n', b'line 2\n']
        for line in lines:
            yield line
            time.sleep(20)

def app(environ, start_response):
    if False:
        for i in range(10):
            print('nop')
    'Application which cooperatively pauses 20 seconds (needed to surpass normal timeouts) before responding'
    status = '200 OK'
    response_headers = [('Content-type', 'text/plain'), ('Transfer-Encoding', 'chunked')]
    sys.stdout.write('request received')
    sys.stdout.flush()
    start_response(status, response_headers)
    return TestIter()