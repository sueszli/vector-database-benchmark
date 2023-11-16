"""
This script serializes the entire traffic dump, including websocket traffic,
as JSON, and either sends it to a URL or writes to a file. The serialization
format is optimized for Elasticsearch; the script can be used to send all
captured traffic to Elasticsearch directly.

Usage:

    mitmproxy
        --mode reverse:http://example.com/
        -s examples/complex/jsondump.py

Configuration:

    Send to a URL:

        cat > ~/.mitmproxy/config.yaml <<EOF
        dump_destination: "https://elastic.search.local/my-index/my-type"
        # Optional Basic auth:
        dump_username: "never-gonna-give-you-up"
        dump_password: "never-gonna-let-you-down"
        # Optional base64 encoding of content fields
        # to store as binary fields in Elasticsearch:
        dump_encodecontent: true
        EOF

    Dump to a local file:

        cat > ~/.mitmproxy/config.yaml <<EOF
        dump_destination: "/user/rastley/output.log"
        EOF
"""
import base64
import json
import logging
from queue import Queue
from threading import Lock
from threading import Thread
import requests
from mitmproxy import ctx
FILE_WORKERS = 1
HTTP_WORKERS = 10

class JSONDumper:
    """
    JSONDumper performs JSON serialization and some extra processing
    for out-of-the-box Elasticsearch support, and then either writes
    the result to a file or sends it to a URL.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self.outfile = None
        self.transformations = None
        self.encode = None
        self.url = None
        self.lock = None
        self.auth = None
        self.queue = Queue()

    def done(self):
        if False:
            print('Hello World!')
        self.queue.join()
        if self.outfile:
            self.outfile.close()
    fields = {'timestamp': (('error', 'timestamp'), ('request', 'timestamp_start'), ('request', 'timestamp_end'), ('response', 'timestamp_start'), ('response', 'timestamp_end'), ('client_conn', 'timestamp_start'), ('client_conn', 'timestamp_end'), ('client_conn', 'timestamp_tls_setup'), ('server_conn', 'timestamp_start'), ('server_conn', 'timestamp_end'), ('server_conn', 'timestamp_tls_setup'), ('server_conn', 'timestamp_tcp_setup')), 'ip': (('server_conn', 'source_address'), ('server_conn', 'ip_address'), ('server_conn', 'address'), ('client_conn', 'address')), 'ws_messages': (('messages',),), 'headers': (('request', 'headers'), ('response', 'headers')), 'content': (('request', 'content'), ('response', 'content'))}

    def _init_transformations(self):
        if False:
            while True:
                i = 10
        self.transformations = [{'fields': self.fields['headers'], 'func': dict}, {'fields': self.fields['timestamp'], 'func': lambda t: int(t * 1000)}, {'fields': self.fields['ip'], 'func': lambda addr: {'host': addr[0].replace('::ffff:', ''), 'port': addr[1]}}, {'fields': self.fields['ws_messages'], 'func': lambda ms: [{'type': m[0], 'from_client': m[1], 'content': base64.b64encode(bytes(m[2], 'utf-8')) if self.encode else m[2], 'timestamp': int(m[3] * 1000)} for m in ms]}]
        if self.encode:
            self.transformations.append({'fields': self.fields['content'], 'func': base64.b64encode})

    @staticmethod
    def transform_field(obj, path, func):
        if False:
            return 10
        '\n        Apply a transformation function `func` to a value\n        under the specified `path` in the `obj` dictionary.\n        '
        for key in path[:-1]:
            if not (key in obj and obj[key]):
                return
            obj = obj[key]
        if path[-1] in obj and obj[path[-1]]:
            obj[path[-1]] = func(obj[path[-1]])

    @classmethod
    def convert_to_strings(cls, obj):
        if False:
            for i in range(10):
                print('nop')
        '\n        Recursively convert all list/dict elements of type `bytes` into strings.\n        '
        if isinstance(obj, dict):
            return {cls.convert_to_strings(key): cls.convert_to_strings(value) for (key, value) in obj.items()}
        elif isinstance(obj, list) or isinstance(obj, tuple):
            return [cls.convert_to_strings(element) for element in obj]
        elif isinstance(obj, bytes):
            return str(obj)[2:-1]
        return obj

    def worker(self):
        if False:
            while True:
                i = 10
        while True:
            frame = self.queue.get()
            self.dump(frame)
            self.queue.task_done()

    def dump(self, frame):
        if False:
            print('Hello World!')
        '\n        Transform and dump (write / send) a data frame.\n        '
        for tfm in self.transformations:
            for field in tfm['fields']:
                self.transform_field(frame, field, tfm['func'])
        frame = self.convert_to_strings(frame)
        if self.outfile:
            self.lock.acquire()
            self.outfile.write(json.dumps(frame) + '\n')
            self.lock.release()
        else:
            requests.post(self.url, json=frame, auth=self.auth or None)

    @staticmethod
    def load(loader):
        if False:
            return 10
        '\n        Extra options to be specified in `~/.mitmproxy/config.yaml`.\n        '
        loader.add_option('dump_encodecontent', bool, False, 'Encode content as base64.')
        loader.add_option('dump_destination', str, 'jsondump.out', 'Output destination: path to a file or URL.')
        loader.add_option('dump_username', str, '', 'Basic auth username for URL destinations.')
        loader.add_option('dump_password', str, '', 'Basic auth password for URL destinations.')

    def configure(self, _):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determine the destination type and path, initialize the output\n        transformation rules.\n        '
        self.encode = ctx.options.dump_encodecontent
        if ctx.options.dump_destination.startswith('http'):
            self.outfile = None
            self.url = ctx.options.dump_destination
            logging.info('Sending all data frames to %s' % self.url)
            if ctx.options.dump_username and ctx.options.dump_password:
                self.auth = (ctx.options.dump_username, ctx.options.dump_password)
                logging.info('HTTP Basic auth enabled.')
        else:
            self.outfile = open(ctx.options.dump_destination, 'a')
            self.url = None
            self.lock = Lock()
            logging.info('Writing all data frames to %s' % ctx.options.dump_destination)
        self._init_transformations()
        for i in range(FILE_WORKERS if self.outfile else HTTP_WORKERS):
            t = Thread(target=self.worker)
            t.daemon = True
            t.start()

    def response(self, flow):
        if False:
            return 10
        '\n        Dump request/response pairs.\n        '
        self.queue.put(flow.get_state())

    def error(self, flow):
        if False:
            i = 10
            return i + 15
        '\n        Dump errors.\n        '
        self.queue.put(flow.get_state())

    def websocket_end(self, flow):
        if False:
            print('Hello World!')
        '\n        Dump websocket messages once the connection ends.\n\n        Alternatively, you can replace `websocket_end` with\n        `websocket_message` if you want the messages to be\n        dumped one at a time with full metadata. Warning:\n        this takes up _a lot_ of space.\n        '
        self.queue.put(flow.get_state())
addons = [JSONDumper()]