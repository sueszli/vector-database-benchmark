import datetime
import json
import logging
from collections import deque
from functools import partial
from ..core.exception import PerspectiveError
from ..table import PerspectiveCppError, Table
from ..table._callback_cache import _PerspectiveCallBackCache
from ..table._date_validator import _PerspectiveDateValidator
_date_validator = _PerspectiveDateValidator()

class DateTimeEncoder(json.JSONEncoder):
    """Before sending datetimes over the wire, convert them to Unix timestamps
    in milliseconds since epoch, using Perspective's date validator to
    ensure consistency."""

    def default(self, obj):
        if False:
            return 10
        if isinstance(obj, datetime.datetime):
            return _date_validator.to_timestamp(obj)
        else:
            return super(DateTimeEncoder, self).default(obj)

class _PerspectiveManagerInternal(object):
    LOCKED_COMMANDS = ['table', 'update', 'remove', 'replace', 'clear']

    def __init__(self, lock=False):
        if False:
            return 10
        self._tables = {}
        self._views = {}
        self._callback_cache = _PerspectiveCallBackCache()
        self._queue_process_callback = None
        self._lock = lock
        self._pending_binary = deque()

    def _get_view(self, name):
        if False:
            i = 10
            return i + 15
        'Return a view under management by name.'
        return self._views.get(name, None)

    def _process(self, msg, post_callback, client_id=None):
        if False:
            for i in range(10):
                print('nop')
        'Given a message from the client, process it through the Perspective\n        engine.\n\n        Args:\n            msg (:obj`dict`): a message from the client with instructions\n                that map to engine operations\n            post_callback (:obj`callable`): a function that returns data to the\n                client. `post_callback` should be a callable that takes two\n                parameters: `data` (str), and `binary` (bool), a kwarg that\n                specifies whether `data` is a binary string.\n        '
        if self._loop_callback:
            self._loop_callback(self.__process, msg, post_callback, client_id)
        else:
            self.__process(msg, post_callback, client_id)

    def __process(self, msg, post_callback, client_id=None):
        if False:
            return 10
        'Process the message through the Perspective engine, handling the\n        special flow for binary messages along the way.'
        if len(self._pending_binary) >= 0 and isinstance(msg, bytes):
            full_message = self._pending_binary.popleft()
            full_message.pop('binary_length')
            new_args = [msg]
            if len(full_message['args']) > 1:
                new_args += full_message['args'][1:]
            full_message['args'] = new_args
            msg = full_message
        if isinstance(msg, str):
            msg = json.loads(msg)
        if not isinstance(msg, dict):
            raise PerspectiveError('Message passed into `_process` should either be a JSON-serialized string or a dict.')
        if msg.get('binary_length', None):
            self._pending_binary.append(msg)
            return
        cmd = msg['cmd']
        if self._is_locked_command(msg) is True:
            error_string = '`{0}` failed - access denied'.format(msg['cmd'] + ('.' + msg['method'] if msg.get('method', None) is not None else ''))
            error_message = self._make_error_message(msg['id'], error_string)
            post_callback(self._message_to_json(msg['id'], error_message))
            return
        try:
            if cmd == 'init':
                flags = ['wait_for_response']
                message = self._make_message(msg['id'], flags)
                post_callback(self._message_to_json(msg['id'], message))
            elif cmd == 'get_hosted_table_names':
                message = self._make_message(msg['id'], [k for k in self._tables.keys()])
                post_callback(self._message_to_json(msg['id'], message))
            elif cmd == 'table':
                try:
                    data_or_schema = msg['args'][0]
                    self._tables[msg['name']] = Table(data_or_schema, **msg.get('options', {}))
                    message = self._make_message(msg['id'], msg['name'])
                    post_callback(self._message_to_json(msg['id'], message))
                except IndexError:
                    self._tables[msg['name']] = []
            elif cmd == 'view':
                new_view = self._tables[msg['table_name']].view(**msg.get('config', {}))
                new_view._client_id = client_id
                self._views[msg['view_name']] = new_view
                message = self._make_message(msg['id'], msg['view_name'])
                post_callback(self._message_to_json(msg['id'], message))
            elif cmd == 'table_method' or cmd == 'view_method':
                self._process_method_call(msg, post_callback, client_id)
            else:
                logging.error('Unknown client message ' + str(msg))
        except (PerspectiveError, PerspectiveCppError) as error:
            error_string = str(error)
            error_message = self._make_error_message(msg['id'], error_string)
            logging.error('[PerspectiveManager._process] %s', error_string)
            post_callback(self._message_to_json(msg['id'], error_message))

    def _process_method_call(self, msg, post_callback, client_id):
        if False:
            print('Hello World!')
        'When the client calls a method, validate the instance it calls on\n        and return the result.\n        '
        if msg['cmd'] == 'table_method':
            table_or_view = self._tables.get(msg['name'], None)
        else:
            table_or_view = self._views.get(msg['name'], None)
            if table_or_view is None:
                error_message = self._make_error_message(msg['id'], 'View method cancelled')
                post_callback(self._message_to_json(msg['id'], error_message))
        try:
            if msg.get('subscribe', False) is True:
                self._process_subscribe(msg, table_or_view, post_callback, client_id)
            else:
                arguments = {}
                if msg['method'] in ('schema', 'expression_schema', 'validate_expressions'):
                    arguments['as_string'] = True
                elif msg['method'].startswith('to_'):
                    for d in msg.get('args', []):
                        arguments.update(d)
                else:
                    arguments = msg.get('args', [])
                if msg['method'] == 'delete':
                    if msg['cmd'] == 'view_method':
                        self._views[msg['name']].delete()
                        self._views.pop(msg['name'], None)
                        return
                    else:
                        raise PerspectiveError('table.delete() cannot be called on a remote table, as the remote has full ownership.')
                if msg['method'].startswith('to_'):
                    result = getattr(table_or_view, msg['method'])(**arguments)
                elif msg['method'] in ('update', 'remove'):
                    data = arguments[0]
                    options = {}
                    if len(arguments) > 1 and isinstance(arguments[1], dict):
                        options = arguments[1]
                    result = getattr(table_or_view, msg['method'])(data, **options)
                elif msg['cmd'] == 'table_method' and msg['method'] == 'validate_expressions':
                    result = getattr(table_or_view, msg['method'])(*msg.get('args', []), **arguments)
                else:
                    result = getattr(table_or_view, msg['method'])(*arguments)
                if isinstance(result, bytes) and msg['method'] != 'to_csv':
                    self._process_bytes(result, msg, post_callback)
                else:
                    message = self._make_message(msg['id'], result)
                    post_callback(self._message_to_json(msg['id'], message))
        except Exception as error:
            error_string = str(error)
            message = self._make_error_message(msg['id'], error_string)
            logging.error('[PerspectiveManager._process_method_call] %s', error_string)
            post_callback(self._message_to_json(msg['id'], message))

    def _process_subscribe(self, msg, table_or_view, post_callback, client_id):
        if False:
            while True:
                i = 10
        'When the client attempts to add or remove a subscription callback,\n        validate and perform the requested operation.\n\n        Args:\n            msg (dict): the message from the client\n            table_or_view {Table|View} : the instance that the subscription\n                will be called on.\n            post_callback (callable): a method that notifies the client with\n                new data.\n            client_id (str) : a unique str id that identifies the\n                `PerspectiveSession` object that is passing the message.\n        '
        try:
            callback = None
            callback_id = msg.get('callback_id', None)
            method = msg.get('method', None)
            args = msg.get('args', [])
            if method and method[:2] == 'on':
                callback = partial(self.callback, msg=msg, post_callback=post_callback)
                if callback_id:
                    self._callback_cache.add_callback({'client_id': client_id, 'callback_id': callback_id, 'callback': callback, 'name': msg.get('name', None)})
            elif callback_id is not None:
                popped_callbacks = self._callback_cache.pop_callbacks(client_id, callback_id)
                for callback in popped_callbacks:
                    getattr(table_or_view, method)(callback['callback'])
            if callback is not None:
                if method == 'on_update':
                    mode = {'mode': 'none'}
                    if len(args) > 0:
                        mode = args[0]
                    getattr(table_or_view, method)(callback, **mode)
                else:
                    getattr(table_or_view, method)(callback)
            else:
                logging.info('callback not found for remote call {}'.format(msg))
        except Exception as error:
            error_string = str(error)
            message = self._make_error_message(msg['id'], error_string)
            logging.error('[PerspectiveManager._process_subscribe] %s', error_string)
            post_callback(self._message_to_json(msg['id'], message))

    def _process_bytes(self, binary, msg, post_callback):
        if False:
            return 10
        "Send a bytestring message to the client without attempting to\n        serialize as JSON.\n\n        Perspective's client expects two messages to be sent when a binary\n        payload is expected: the first message is a JSON-serialized string with\n        the message's `id` and `msg`, and the second message is a bytestring\n        without any additional metadata. Implementations of the `post_callback`\n        should have an optional kwarg named `binary`, which specifies whether\n        `data` is a bytestring or not.\n\n        Args:\n            binary (bytes, bytearray) : a byte message to be passed to the client.\n            msg (dict) : the original message that generated the binary\n                response from Perspective.\n            post_callback (callable) : a function that passes data to the\n                client, with a `binary` (bool) kwarg that allows it to pass\n                byte messages without serializing to JSON.\n        "
        msg['binary_length'] = len(binary)
        post_callback(json.dumps(msg, cls=DateTimeEncoder))
        post_callback(binary, binary=True)

    def callback(self, *args, **kwargs):
        if False:
            return 10
        'Return a message to the client using the `post_callback` method.'
        orig_msg = kwargs.get('msg')
        id = orig_msg['id']
        method = orig_msg['method']
        post_callback = kwargs.get('post_callback')
        if method == 'on_update':
            updated = {'port_id': args[0]}
            msg = self._make_message(id, updated)
        else:
            msg = self._make_message(id, None)
        if len(args) > 1 and type(args[1]) is bytes:
            self._process_bytes(args[1], msg, post_callback)
        else:
            post_callback(self._message_to_json(msg['id'], msg))

    def clear_views(self, client_id):
        if False:
            return 10
        'Garbage collect views that belong to closed connections.'
        count = 0
        names = []
        if not client_id:
            raise PerspectiveError('Cannot garbage collect views that are not linked to a specific client ID!')
        for (name, view) in self._views.items():
            if view._client_id == client_id:
                if self._loop_callback:
                    self._loop_callback(view.delete)
                else:
                    view.delete()
                names.append(name)
                count += 1
        for name in names:
            self._views.pop(name)
        if count > 0:
            logging.debug('client {} disconnected - GC {} views in memory'.format(client_id, count))

    def _make_message(self, id, result):
        if False:
            return 10
        'Return a serializable message for a successful result.'
        return {'id': id, 'data': result}

    def _make_error_message(self, id, error):
        if False:
            return 10
        'Return a serializable message for an error result.'
        return {'id': id, 'error': error}

    def _message_to_json(self, id, message):
        if False:
            return 10
        'Given a message object to be passed to Perspective, serialize it\n        into a string using `DateTimeEncoder` and `allow_nan=False`.\n\n        If an Exception occurs in serialization, catch the Exception and\n        return an error message using `self._make_error_message`.\n\n        Args:\n            message (:obj:`dict`) a serializable message to be passed to\n                Perspective.\n        '
        try:
            return json.dumps(message, allow_nan=False, cls=DateTimeEncoder)
        except ValueError as error:
            error_string = str(error)
            if error_string == 'Out of range float values are not JSON compliant':
                error_string = 'Cannot serialize `NaN`, `Infinity` or `-Infinity` to JSON.'
            error_message = self._make_error_message(id, 'JSON serialization error: {}'.format(error_string))
            logging.warning(error_message['error'])
            return json.dumps(error_message)

    def _is_locked_command(self, msg):
        if False:
            while True:
                i = 10
        'Returns `True` if the manager instance is locked and the command\n        is in `_PerspectiveManagerInternal.LOCKED_COMMANDS`, and `False` otherwise.'
        if not self._lock:
            return False
        cmd = msg['cmd']
        method = msg.get('method', None)
        if cmd == 'table_method' and method == 'delete':
            return True
        return cmd == 'table' or method in _PerspectiveManagerInternal.LOCKED_COMMANDS