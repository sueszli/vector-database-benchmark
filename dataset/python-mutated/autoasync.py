from asyncio import get_event_loop, iscoroutine
from functools import wraps
from inspect import signature

def _launch_forever_coro(coro, args, kwargs, loop):
    if False:
        i = 10
        return i + 15
    '\n    This helper function launches an async main function that was tagged with\n    forever=True. There are two possibilities:\n\n    - The function is a normal function, which handles initializing the event\n      loop, which is then run forever\n    - The function is a coroutine, which needs to be scheduled in the event\n      loop, which is then run forever\n      - There is also the possibility that the function is a normal function\n        wrapping a coroutine function\n\n    The function is therefore called unconditionally and scheduled in the event\n    loop if the return value is a coroutine object.\n\n    The reason this is a separate function is to make absolutely sure that all\n    the objects created are garbage collected after all is said and done; we\n    do this to ensure that any exceptions raised in the tasks are collected\n    ASAP.\n    '
    thing = coro(*args, **kwargs)
    if iscoroutine(thing):
        loop.create_task(thing)

def autoasync(coro=None, *, loop=None, forever=False, pass_loop=False):
    if False:
        print('Hello World!')
    "\n    Convert an asyncio coroutine into a function which, when called, is\n    evaluted in an event loop, and the return value returned. This is intented\n    to make it easy to write entry points into asyncio coroutines, which\n    otherwise need to be explictly evaluted with an event loop's\n    run_until_complete.\n\n    If `loop` is given, it is used as the event loop to run the coro in. If it\n    is None (the default), the loop is retreived using asyncio.get_event_loop.\n    This call is defered until the decorated function is called, so that\n    callers can install custom event loops or event loop policies after\n    @autoasync is applied.\n\n    If `forever` is True, the loop is run forever after the decorated coroutine\n    is finished. Use this for servers created with asyncio.start_server and the\n    like.\n\n    If `pass_loop` is True, the event loop object is passed into the coroutine\n    as the `loop` kwarg when the wrapper function is called. In this case, the\n    wrapper function's __signature__ is updated to remove this parameter, so\n    that autoparse can still be used on it without generating a parameter for\n    `loop`.\n\n    This coroutine can be called with ( @autoasync(...) ) or without\n    ( @autoasync ) arguments.\n\n    Examples:\n\n    @autoasync\n    def get_file(host, port):\n        reader, writer = yield from asyncio.open_connection(host, port)\n        data = reader.read()\n        sys.stdout.write(data.decode())\n\n    get_file(host, port)\n\n    @autoasync(forever=True, pass_loop=True)\n    def server(host, port, loop):\n        yield_from loop.create_server(Proto, host, port)\n\n    server('localhost', 8899)\n\n    "
    if coro is None:
        return lambda c: autoasync(c, loop=loop, forever=forever, pass_loop=pass_loop)
    if pass_loop:
        old_sig = signature(coro)
        new_sig = old_sig.replace(parameters=(param for (name, param) in old_sig.parameters.items() if name != 'loop'))

    @wraps(coro)
    def autoasync_wrapper(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        local_loop = get_event_loop() if loop is None else loop
        if pass_loop:
            bound_args = old_sig.bind_partial()
            bound_args.arguments.update(loop=local_loop, **new_sig.bind(*args, **kwargs).arguments)
            (args, kwargs) = (bound_args.args, bound_args.kwargs)
        if forever:
            _launch_forever_coro(coro, args, kwargs, local_loop)
            local_loop.run_forever()
        else:
            return local_loop.run_until_complete(coro(*args, **kwargs))
    if pass_loop:
        autoasync_wrapper.__signature__ = new_sig
    return autoasync_wrapper