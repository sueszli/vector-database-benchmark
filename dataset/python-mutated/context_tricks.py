@object.__new__
class nop_context(object):
    """A nop context manager.
    """

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        pass

    def __exit__(self, *excinfo):
        if False:
            while True:
                i = 10
        pass

def _nop(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    pass

class CallbackManager(object):
    """Create a context manager from a pre-execution callback and a
    post-execution callback.

    Parameters
    ----------
    pre : (...) -> any, optional
        A pre-execution callback. This will be passed ``*args`` and
        ``**kwargs``.
    post : (...) -> any, optional
        A post-execution callback. This will be passed ``*args`` and
        ``**kwargs``.

    Notes
    -----
    The enter value of this context manager will be the result of calling
    ``pre(*args, **kwargs)``

    Examples
    --------
    >>> def pre(where):
    ...     print('entering %s block' % where)
    >>> def post(where):
    ...     print('exiting %s block' % where)
    >>> manager = CallbackManager(pre, post)
    >>> with manager('example'):
    ...    print('inside example block')
    entering example block
    inside example block
    exiting example block

    These are reusable with different args:
    >>> with manager('another'):
    ...     print('inside another block')
    entering another block
    inside another block
    exiting another block
    """

    def __init__(self, pre=None, post=None):
        if False:
            for i in range(10):
                print('nop')
        self.pre = pre if pre is not None else _nop
        self.post = post if post is not None else _nop

    def __call__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        return _ManagedCallbackContext(self.pre, self.post, args, kwargs)

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        return self.pre()

    def __exit__(self, *excinfo):
        if False:
            i = 10
            return i + 15
        self.post()

class _ManagedCallbackContext(object):

    def __init__(self, pre, post, args, kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._pre = pre
        self._post = post
        self._args = args
        self._kwargs = kwargs

    def __enter__(self):
        if False:
            while True:
                i = 10
        return self._pre(*self._args, **self._kwargs)

    def __exit__(self, *excinfo):
        if False:
            return 10
        self._post(*self._args, **self._kwargs)