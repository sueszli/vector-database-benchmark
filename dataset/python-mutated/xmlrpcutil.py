"""XML-RPC tool helpers."""
import sys
from xmlrpc.client import loads as xmlrpc_loads, dumps as xmlrpc_dumps, Fault as XMLRPCFault
import cherrypy
from cherrypy._cpcompat import ntob

def process_body():
    if False:
        i = 10
        return i + 15
    'Return (params, method) from request body.'
    try:
        return xmlrpc_loads(cherrypy.request.body.read())
    except Exception:
        return (('ERROR PARAMS',), 'ERRORMETHOD')

def patched_path(path):
    if False:
        print('Hello World!')
    "Return 'path', doctored for RPC."
    if not path.endswith('/'):
        path += '/'
    if path.startswith('/RPC2/'):
        path = path[5:]
    return path

def _set_response(body):
    if False:
        while True:
            i = 10
    'Set up HTTP status, headers and body within CherryPy.'
    response = cherrypy.response
    response.status = '200 OK'
    response.body = ntob(body, 'utf-8')
    response.headers['Content-Type'] = 'text/xml'
    response.headers['Content-Length'] = len(body)

def respond(body, encoding='utf-8', allow_none=0):
    if False:
        while True:
            i = 10
    'Construct HTTP response body.'
    if not isinstance(body, XMLRPCFault):
        body = (body,)
    _set_response(xmlrpc_dumps(body, methodresponse=1, encoding=encoding, allow_none=allow_none))

def on_error(*args, **kwargs):
    if False:
        return 10
    'Construct HTTP response body for an error response.'
    body = str(sys.exc_info()[1])
    _set_response(xmlrpc_dumps(XMLRPCFault(1, body)))