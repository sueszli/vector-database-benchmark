import logging
import json
import vim
from vimspector import utils
DEFAULT_SYNC_TIMEOUT = 5000
DEFAULT_ASYNC_TIMEOUT = 15000

class PendingRequest(object):

    def __init__(self, msg, handler, failure_handler, expiry_id):
        if False:
            print('Hello World!')
        self.msg = msg
        self.handler = handler
        self.failure_handler = failure_handler
        self.expiry_id = expiry_id

class DebugAdapterConnection(object):

    def __init__(self, handlers, session_id, send_func, sync_timeout=None, async_timeout=None):
        if False:
            print('Hello World!')
        self._logger = logging.getLogger(__name__ + '.' + str(session_id))
        utils.SetUpLogging(self._logger, session_id)
        if not sync_timeout:
            sync_timeout = DEFAULT_SYNC_TIMEOUT
        if not async_timeout:
            async_timeout = DEFAULT_ASYNC_TIMEOUT
        self._Write = send_func
        self._SetState('READ_HEADER')
        self._buffer = bytes()
        self._handlers = handlers
        self._session_id = session_id
        self._next_message_id = 0
        self._outstanding_requests = {}
        self.async_timeout = async_timeout
        self.sync_timeout = sync_timeout

    def GetSessionId(self):
        if False:
            print('Hello World!')
        return self._session_id

    def DoRequest(self, handler, msg, failure_handler=None, timeout=None):
        if False:
            while True:
                i = 10
        if timeout is None:
            timeout = self.async_timeout
        this_id = self._next_message_id
        self._next_message_id += 1
        msg['seq'] = this_id
        msg['type'] = 'request'
        expiry_id = vim.eval('timer_start( {},              function( "vimspector#internal#channel#Timeout",                        [ {} ] ) )'.format(timeout, self._session_id))
        request = PendingRequest(msg, handler, failure_handler, expiry_id)
        self._outstanding_requests[this_id] = request
        if not self._SendMessage(msg):
            self._AbortRequest(request, 'Unable to send message')

    def DoRequestSync(self, msg, timeout=None):
        if False:
            while True:
                i = 10
        result = {}
        if timeout is None:
            timeout = self.sync_timeout

        def handler(msg):
            if False:
                for i in range(10):
                    print('nop')
            result['response'] = msg

        def failure_handler(reason, msg):
            if False:
                print('Hello World!')
            result['response'] = msg
            result['exception'] = RuntimeError(reason)
        self.DoRequest(handler, msg, failure_handler, timeout)
        to_wait = timeout + 1000
        while not result and to_wait >= 0:
            vim.command('sleep 10m')
            to_wait -= 10
        if result.get('exception') is not None:
            raise result['exception']
        if result.get('response') is None:
            raise RuntimeError('No response')
        return result['response']

    def OnRequestTimeout(self, timer_id):
        if False:
            print('Hello World!')
        request_id = None
        for (seq, request) in self._outstanding_requests.items():
            if request.expiry_id == timer_id:
                request_id = seq
                break
        if request_id is not None:
            request = self._outstanding_requests.pop(request_id)
            self._AbortRequest(request, 'Timeout')

    def DoResponse(self, request, error, response):
        if False:
            while True:
                i = 10
        this_id = self._next_message_id
        self._next_message_id += 1
        msg = {}
        msg['seq'] = this_id
        msg['type'] = 'response'
        msg['request_seq'] = request['seq']
        msg['command'] = request['command']
        msg['body'] = response
        if error:
            msg['success'] = False
            msg['message'] = error
        else:
            msg['success'] = True
        self._SendMessage(msg)

    def Reset(self):
        if False:
            print('Hello World!')
        self._Write = None
        self._handlers = None
        while self._outstanding_requests:
            (_, request) = self._outstanding_requests.popitem()
            self._AbortRequest(request, 'Closing down')

    def _AbortRequest(self, request, reason):
        if False:
            return 10
        self._logger.debug('{}: Aborting request {}'.format(reason, request.msg))
        _KillTimer(request)
        if request.failure_handler:
            request.failure_handler(reason, {})
        else:
            utils.UserMessage('Request for {} aborted: {}'.format(request.msg['command'], reason))

    def OnData(self, data):
        if False:
            i = 10
            return i + 15
        data = bytes(data, 'utf-8')
        self._buffer += data
        while True:
            if self._state == 'READ_HEADER':
                self._ReadHeaders()
            if self._state == 'READ_BODY':
                self._ReadBody()
            else:
                break
            if self._state != 'READ_HEADER':
                break

    def _SetState(self, state):
        if False:
            return 10
        self._state = state
        if state == 'READ_HEADER':
            self._headers = {}

    def _SendMessage(self, msg):
        if False:
            return 10
        if not self._Write:
            return False
        msg = json.dumps(msg)
        self._logger.debug('Sending Message: {0}'.format(msg))
        data = 'Content-Length: {0}\r\n\r\n{1}'.format(len(msg), msg)
        return self._Write(data)

    def _ReadHeaders(self):
        if False:
            print('Hello World!')
        parts = self._buffer.split(bytes('\r\n\r\n', 'utf-8'), 1)
        if len(parts) > 1:
            headers = parts[0]
            for header_line in headers.split(bytes('\r\n', 'utf-8')):
                if bytes('\n', 'utf-8') in header_line:
                    header_line = header_line.split(bytes('\n', 'utf-8'))[-1]
                if header_line.strip():
                    (key, value) = str(header_line, 'utf-8').split(':', 1)
                    self._headers[key] = value
            self._buffer = parts[1]
            self._SetState('READ_BODY')
            return

    def _ReadBody(self):
        if False:
            return 10
        try:
            content_length = int(self._headers['Content-Length'])
        except KeyError:
            self._logger.error('Missing Content-Length header in: {0}'.format(json.dumps(self._headers)))
            self._buffer = bytes('', 'utf-8')
            self._SetState('READ_HEADER')
            return
        if len(self._buffer) < content_length:
            assert self._state == 'READ_BODY'
            return
        payload = str(self._buffer[:content_length], 'utf-8')
        self._buffer = self._buffer[content_length:]
        self._SetState('READ_HEADER')
        try:
            message = json.loads(payload, strict=False)
        except Exception:
            self._logger.exception('Invalid message received: %s', payload)
            raise
        self._logger.debug('Message received: {0}'.format(message))
        self._OnMessageReceived(message)

    def _OnMessageReceived(self, message):
        if False:
            return 10
        if not self._handlers:
            return
        if message['type'] == 'response':
            try:
                request = self._outstanding_requests.pop(message['request_seq'])
            except KeyError:
                utils.UserMessage('Protocol error: duplicate response for request {}'.format(message['request_seq']))
                self._logger.exception('Duplicate response: {}'.format(message))
                return
            _KillTimer(request)
            if message['success']:
                if request.handler:
                    request.handler(message)
            else:
                reason = message.get('message')
                error = message.get('body', {}).get('error', {})
                if error:
                    try:
                        fmt = error['format']
                        variables = error.get('variables', {})
                        reason = fmt.format(**variables)
                    except Exception:
                        self._logger.exception('Failed to parse error, using default: %s', error)
                if request.failure_handler:
                    self._logger.info('Request failed (handled): %s', reason)
                    request.failure_handler(reason, message)
                else:
                    self._logger.error('Request failed (unhandled): %s', reason)
                    for h in self._handlers:
                        if 'OnFailure' in dir(h):
                            if h.OnFailure(reason, request.msg, message):
                                break
        elif message['type'] == 'event':
            method = 'OnEvent_' + message['event']
            for h in self._handlers:
                if method in dir(h):
                    if getattr(h, method)(message):
                        break
        elif message['type'] == 'request':
            method = 'OnRequest_' + message['command']
            for h in self._handlers:
                if method in dir(h):
                    if getattr(h, method)(message):
                        break

def _KillTimer(request):
    if False:
        while True:
            i = 10
    if request.expiry_id is not None:
        vim.eval('timer_stop( {} )'.format(request.expiry_id))
        request.expiry_id = None