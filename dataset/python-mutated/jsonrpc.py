from __future__ import annotations
import json
import pickle
import traceback
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.six import binary_type, text_type
from ansible.utils.display import Display
display = Display()

class JsonRpcServer(object):
    _objects = set()

    def handle_request(self, request):
        if False:
            for i in range(10):
                print('nop')
        request = json.loads(to_text(request, errors='surrogate_then_replace'))
        method = request.get('method')
        if method.startswith('rpc.') or method.startswith('_'):
            error = self.invalid_request()
            return json.dumps(error)
        (args, kwargs) = request.get('params')
        setattr(self, '_identifier', request.get('id'))
        rpc_method = None
        for obj in self._objects:
            rpc_method = getattr(obj, method, None)
            if rpc_method:
                break
        if not rpc_method:
            error = self.method_not_found()
            response = json.dumps(error)
        else:
            try:
                result = rpc_method(*args, **kwargs)
            except ConnectionError as exc:
                display.vvv(traceback.format_exc())
                try:
                    error = self.error(code=exc.code, message=to_text(exc))
                except AttributeError:
                    error = self.internal_error(data=to_text(exc))
                response = json.dumps(error)
            except Exception as exc:
                display.vvv(traceback.format_exc())
                error = self.internal_error(data=to_text(exc, errors='surrogate_then_replace'))
                response = json.dumps(error)
            else:
                if isinstance(result, dict) and 'jsonrpc' in result:
                    response = result
                else:
                    response = self.response(result)
                try:
                    response = json.dumps(response)
                except Exception as exc:
                    display.vvv(traceback.format_exc())
                    error = self.internal_error(data=to_text(exc, errors='surrogate_then_replace'))
                    response = json.dumps(error)
        delattr(self, '_identifier')
        return response

    def register(self, obj):
        if False:
            while True:
                i = 10
        self._objects.add(obj)

    def header(self):
        if False:
            print('Hello World!')
        return {'jsonrpc': '2.0', 'id': self._identifier}

    def response(self, result=None):
        if False:
            for i in range(10):
                print('nop')
        response = self.header()
        if isinstance(result, binary_type):
            result = to_text(result)
        if not isinstance(result, text_type):
            response['result_type'] = 'pickle'
            result = to_text(pickle.dumps(result, protocol=0))
        response['result'] = result
        return response

    def error(self, code, message, data=None):
        if False:
            print('Hello World!')
        response = self.header()
        error = {'code': code, 'message': message}
        if data:
            error['data'] = data
        response['error'] = error
        return response

    def parse_error(self, data=None):
        if False:
            return 10
        return self.error(-32700, 'Parse error', data)

    def method_not_found(self, data=None):
        if False:
            return 10
        return self.error(-32601, 'Method not found', data)

    def invalid_request(self, data=None):
        if False:
            return 10
        return self.error(-32600, 'Invalid request', data)

    def invalid_params(self, data=None):
        if False:
            i = 10
            return i + 15
        return self.error(-32602, 'Invalid params', data)

    def internal_error(self, data=None):
        if False:
            for i in range(10):
                print('nop')
        return self.error(-32603, 'Internal error', data)