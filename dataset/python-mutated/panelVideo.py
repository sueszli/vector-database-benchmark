import os, sys, re
import mimetypes
import public
from BTPanel import Response

def get_buff_size(file_size):
    if False:
        return 10
    buff_size = 2097152
    if file_size > 1073741824:
        buff_size = 4194304
    return buff_size

def partial_response(path, start, end=None):
    if False:
        print('Hello World!')
    if not os.path.exists(path):
        return Response('file not fount!', 404)
    file_size = os.path.getsize(path)
    buff_size = get_buff_size(file_size)
    if end is None:
        end = start + buff_size - 1
    end = min(end, file_size - 1)
    end = min(end, start + buff_size - 1)
    length = end - start + 1
    with open(path, 'rb') as fd:
        fd.seek(start)
        bytes = fd.read(length)
    assert len(bytes) == length
    response = Response(bytes, 206, mimetype=mimetypes.guess_type(path)[0], direct_passthrough=True)
    response.headers.add('Content-Range', 'bytes {0}-{1}/{2}'.format(start, end, file_size))
    response.headers.add('Accept-Ranges', 'bytes')
    return response

def get_range(request):
    if False:
        print('Hello World!')
    range = request.headers.get('Range')
    m = None
    if range:
        m = re.match('bytes=(?P<start>\\d+)-(?P<end>\\d+)?', range)
    if m:
        start = m.group('start')
        end = m.group('end')
        start = int(start)
        if end is not None:
            end = int(end)
        return (start, end)
    else:
        return (0, None)