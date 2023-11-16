import logging
import sys
from collections import defaultdict
from itertools import chain, islice, repeat
from logging import StreamHandler
from types import SimpleNamespace
ENCODING = sys.getdefaultencoding()

def buffered_hook_manager(header_template, get_pos, cond_refresh, term):
    if False:
        for i in range(10):
            print('nop')
    'Create and maintain a buffered hook manager, used for instrumenting print\n    statements and logging.\n\n    Args:\n        header_template (): the template for enriching output\n        get_pos (Callable[..., Any]): the container to retrieve the current position\n        cond_refresh: Condition object to force a refresh when printing\n        term: the current terminal\n\n    Returns:\n        a closure with several functions\n\n    '

    def flush_buffers():
        if False:
            return 10
        for (stream, buffer) in buffers.items():
            flush(stream)

    def flush(stream):
        if False:
            i = 10
            return i + 15
        if buffers[stream]:
            write(stream, '\n')
            stream.flush()

    def write(stream, part):
        if False:
            while True:
                i = 10
        if isinstance(part, bytes):
            part = part.decode(ENCODING)
        buffer = buffers[stream]
        if part != '\n':
            osc = part.find('\x1b]')
            if osc >= 0:
                (end, s) = (part.find('\x07', osc + 2), 1)
                if end < 0:
                    (end, s) = (part.find('\x1b\\', osc + 2), 2)
                    if end < 0:
                        (end, s) = (len(part), 0)
                stream.write(part[osc:end + s])
                stream.flush()
                part = part[:osc] + part[end + s:]
                if not part:
                    return
            gen = chain.from_iterable(zip(repeat(None), part.splitlines(True)))
            buffer.extend(islice(gen, 1, None))
        else:
            header = get_header()
            spacer = ' ' * len(header)
            nested = ''.join((line or spacer for line in buffer))
            text = f'{header}{nested.rstrip()}\n'
            with cond_refresh:
                if stream in base:
                    term.clear_line()
                    term.clear_end_screen()
                stream.write(text)
                stream.flush()
                cond_refresh.notify()
                buffer[:] = []

    class Hook(BaseHook):

        def write(self, part):
            if False:
                return 10
            return write(self._stream, part)

        def flush(self):
            if False:
                for i in range(10):
                    print('nop')
            return flush(self._stream)

    def get_hook_for(handler):
        if False:
            return 10
        if handler.stream:
            handler.stream.flush()
        return Hook(handler.stream)

    def install():
        if False:
            while True:
                i = 10

        def get_all_loggers():
            if False:
                return 10
            yield logging.root
            yield from (logging.getLogger(name) for name in logging.root.manager.loggerDict)

        def set_hook(h):
            if False:
                print('Hello World!')
            try:
                return h.setStream(get_hook_for(h))
            except Exception:
                pass
        handlers = set((h for logger in get_all_loggers() for h in logger.handlers if isinstance(h, StreamHandler)))
        before_handlers.update({h: set_hook(h) for h in handlers})
        (sys.stdout, sys.stderr) = (get_hook_for(SimpleNamespace(stream=x)) for x in base)

    def uninstall():
        if False:
            return 10
        flush_buffers()
        buffers.clear()
        (sys.stdout, sys.stderr) = base
        [handler.setStream(original) for (handler, original) in before_handlers.items() if original]
        before_handlers.clear()
    if issubclass(sys.stdout.__class__, BaseHook):
        raise UserWarning('Nested use of alive_progress is not yet supported.')
    buffers = defaultdict(list)
    get_header = gen_header(header_template, get_pos) if header_template else null_header
    base = (sys.stdout, sys.stderr)
    before_handlers = {}
    hook_manager = SimpleNamespace(flush_buffers=flush_buffers, install=install, uninstall=uninstall)
    return hook_manager

class BaseHook:

    def __init__(self, stream):
        if False:
            while True:
                i = 10
        self._stream = stream

    def __getattr__(self, item):
        if False:
            return 10
        return getattr(self._stream, item)

def passthrough_hook_manager():
    if False:
        return 10
    passthrough_hook_manager.flush_buffers = __noop
    passthrough_hook_manager.install = __noop
    passthrough_hook_manager.uninstall = __noop
    return passthrough_hook_manager

def __noop():
    if False:
        for i in range(10):
            print('nop')
    pass

def gen_header(header_template, get_pos):
    if False:
        i = 10
        return i + 15

    def inner():
        if False:
            while True:
                i = 10
        return header_template.format(get_pos())
    return inner

def null_header():
    if False:
        i = 10
        return i + 15
    return ''