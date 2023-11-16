"""
Copyright 2008-2015 Free Software Foundation, Inc.
This file is part of GNU Radio

SPDX-License-Identifier: GPL-2.0-or-later

"""
import sys
import re
import subprocess
import threading
import json
import random
import itertools
import queue

def docstring_guess_from_key(key):
    if False:
        return 10
    '\n    Extract the documentation from the python __doc__ strings\n    By guessing module and constructor names from key\n\n    Args:\n        key: the block key\n\n    Returns:\n        a dict (block_name --> doc string)\n    '
    doc_strings = dict()
    in_tree = [key.partition('_')[::2] + (lambda package: getattr(__import__('gnuradio.' + package), package),)]
    key_parts = key.split('_')
    oot = [('_'.join(key_parts[:i]), '_'.join(key_parts[i:]), __import__) for i in range(1, len(key_parts))]
    for (module_name, init_name, importer) in itertools.chain(in_tree, oot):
        if not module_name or not init_name:
            continue
        try:
            module = importer(module_name)
            break
        except ImportError:
            continue
    else:
        return doc_strings
    pattern = re.compile('^' + init_name.replace('_', '_*').replace('x', '\\w') + '\\w*$')
    for match in filter(pattern.match, dir(module)):
        try:
            doc_strings[match] = getattr(module, match).__doc__
        except AttributeError:
            continue
    return doc_strings

def docstring_from_make(key, imports, make):
    if False:
        for i in range(10):
            print('nop')
    '\n    Extract the documentation from the python __doc__ strings\n    By importing it and checking a truncated make\n\n    Args:\n        key: the block key\n        imports: a list of import statements (string) to execute\n        make: block constructor template\n\n    Returns:\n        a list of tuples (block_name, doc string)\n    '
    try:
        blk_cls = make.partition('(')[0].strip()
        if '$' in blk_cls:
            raise ValueError('Not an identifier')
        ns = dict()
        exec(imports.strip(), ns)
        blk = eval(blk_cls, ns)
        doc_strings = {key: blk.__doc__}
    except (ImportError, AttributeError, SyntaxError, ValueError):
        doc_strings = docstring_guess_from_key(key)
    return doc_strings

class SubprocessLoader(object):
    """
    Start and manage docstring extraction process
    Manages subprocess and handles RPC.
    """
    BOOTSTRAP = "import runpy; runpy.run_path({!r}, run_name='__worker__')"
    AUTH_CODE = random.random()
    RESTART = 5
    DONE = object()

    def __init__(self, callback_query_result, callback_finished=None):
        if False:
            i = 10
            return i + 15
        self.callback_query_result = callback_query_result
        self.callback_finished = callback_finished or (lambda : None)
        self._queue = queue.Queue()
        self._thread = None
        self._worker = None
        self._shutdown = threading.Event()
        self._last_cmd = None

    def start(self):
        if False:
            i = 10
            return i + 15
        ' Start the worker process handler thread '
        if self._thread is not None:
            return
        self._shutdown.clear()
        thread = self._thread = threading.Thread(target=self.run_worker)
        thread.daemon = True
        thread.start()

    def run_worker(self):
        if False:
            i = 10
            return i + 15
        ' Read docstring back from worker stdout and execute callback. '
        for _ in range(self.RESTART):
            if self._shutdown.is_set():
                break
            try:
                self._worker = subprocess.Popen(args=(sys.executable, '-uc', self.BOOTSTRAP.format(__file__)), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self._handle_worker()
            except (OSError, IOError):
                msg = 'Warning: restarting the docstring loader'
                (cmd, args) = self._last_cmd
                if cmd == 'query':
                    msg += ' (crashed while loading {0!r})'.format(args[0])
                print(msg, file=sys.stderr)
                continue
            else:
                break
            finally:
                if self._worker:
                    self._worker.stdin.close()
                    self._worker.stdout.close()
                    self._worker.stderr.close()
                    self._worker.terminate()
                    self._worker.wait()
        else:
            print('Warning: docstring loader crashed too often', file=sys.stderr)
        self._thread = None
        self._worker = None
        self.callback_finished()

    def _handle_worker(self):
        if False:
            for i in range(10):
                print('nop')
        ' Send commands and responses back from worker. '
        assert '1' == self._worker.stdout.read(1).decode('utf-8')
        for (cmd, args) in iter(self._queue.get, self.DONE):
            self._last_cmd = (cmd, args)
            self._send(cmd, args)
            (cmd, args) = self._receive()
            self._handle_response(cmd, args)

    def _send(self, cmd, args):
        if False:
            print('Hello World!')
        " Send a command to worker's stdin "
        fd = self._worker.stdin
        query = json.dumps((self.AUTH_CODE, cmd, args))
        fd.write(query.encode('utf-8'))
        fd.write(b'\n')
        fd.flush()

    def _receive(self):
        if False:
            for i in range(10):
                print('nop')
        " Receive response from worker's stdout "
        for line in iter(self._worker.stdout.readline, ''):
            try:
                (key, cmd, args) = json.loads(line.decode('utf-8'))
                if key != self.AUTH_CODE:
                    raise ValueError('Got wrong auth code')
                return (cmd, args)
            except ValueError:
                if self._worker.poll():
                    raise IOError('Worker died')
                else:
                    continue
        else:
            raise IOError("Can't read worker response")

    def _handle_response(self, cmd, args):
        if False:
            while True:
                i = 10
        ' Handle response from worker, call the callback '
        if cmd == 'result':
            (key, docs) = args
            self.callback_query_result(key, docs)
        elif cmd == 'error':
            print(args)
        else:
            print('Unknown response:', cmd, args, file=sys.stderr)

    def query(self, key, imports=None, make=None):
        if False:
            return 10
        ' Request docstring extraction for a certain key '
        if self._thread is None:
            self.start()
        if imports and make:
            self._queue.put(('query', (key, imports, make)))
        else:
            self._queue.put(('query_key_only', (key,)))

    def finish(self):
        if False:
            print('Hello World!')
        ' Signal end of requests '
        self._queue.put(self.DONE)

    def wait(self):
        if False:
            return 10
        ' Wait for the handler thread to die '
        if self._thread:
            self._thread.join()

    def terminate(self):
        if False:
            return 10
        ' Terminate the worker and wait '
        self._shutdown.set()
        try:
            self._worker.terminate()
            self.wait()
        except (OSError, AttributeError):
            pass

def worker_main():
    if False:
        return 10
    '\n    Main entry point for the docstring extraction process.\n    Manages RPC with main process through stdin/stdout.\n    Runs a docstring extraction for each key it read on stdin.\n    '

    def send(code, cmd, args):
        if False:
            while True:
                i = 10
        json.dump((code, cmd, args), sys.stdout)
        sys.stdout.write('\n')
        sys.stdout.flush()
    sys.stdout.write('1')
    sys.stdout.flush()
    for line in iter(sys.stdin.readline, ''):
        (code, cmd, args) = json.loads(line)
        try:
            if cmd == 'query':
                (key, imports, make) = args
                send(code, 'result', (key, docstring_from_make(key, imports, make)))
            elif cmd == 'query_key_only':
                (key,) = args
                send(code, 'result', (key, docstring_guess_from_key(key)))
            elif cmd == 'exit':
                break
        except Exception as e:
            send(code, 'error', repr(e))
if __name__ == '__worker__':
    worker_main()
elif __name__ == '__main__':

    def callback(key, docs):
        if False:
            i = 10
            return i + 15
        print(key)
        for (match, doc) in docs.items():
            print('-->', match)
            print(str(doc).strip())
            print()
        print()
    r = SubprocessLoader(callback)
    r.query('expr_utils_graph')
    r.query('blocks_add_cc')
    r.query('blocks_add_cc', ['import gnuradio.blocks'], 'gnuradio.blocks.add_cc(')
    r.finish()
    r.wait()