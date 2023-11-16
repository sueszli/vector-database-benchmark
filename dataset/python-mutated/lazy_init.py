from ...base import framework
__all__ = ['LazyGuard']

class LazyInitHelper:
    """
    A Helper Context to trigger switching mode between dygraph and static graph mode,
    and holds the startup program resource.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self._state = False
        self._tracer = None
        self._in_guard = False

    def enable(self):
        if False:
            i = 10
            return i + 15
        '\n        Switch into lazy mode.\n\n        NOTE(dev): This is a very low level API and not exposed for user.\n        '
        if self._state:
            return
        assert framework.in_dygraph_mode(), 'LazyInit.enable() is only available in dygraph mode.'
        self._state = True

    def disable(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Exit from lazy mode.\n\n        NOTE(dev): This is a very low level API and not exposed for user.\n        '
        if not self._state:
            return
        self._state = False

    def __enter__(self):
        if False:
            print('Hello World!')
        '\n        Switch into lazy mode and set _dygraph_tracer_ with None to convert\n        dygraph mode into static graph mode.\n        '
        self.enable()
        if self._in_guard:
            return
        self._tracer = framework.global_var._dygraph_tracer_
        framework.global_var._dygraph_tracer_ = None
        self._in_guard = True

    def __exit__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Exit from lazy mode and recover _dygraph_tracer_.\n        '
        self.disable()
        if not self._in_guard:
            return
        assert self._tracer is not None
        framework.global_var._dygraph_tracer_ = self._tracer
        self._tracer = None
        self._in_guard = False

    @property
    def state(self):
        if False:
            i = 10
            return i + 15
        return self._state
_lazy_init_helper = LazyInitHelper()

def lazy_init_helper():
    if False:
        return 10
    global _lazy_init_helper
    return _lazy_init_helper

class LazyGuard:
    """
    LazyGuard is a wrapper interface for nn.Layer, it forwards the construct
    process of user defined Layer. Meanwhile, it provides necessary API to
    trigger EagerParamBase Lazy Initialization and get startup Program.

    Examples:

        .. code-block:: python

            >>> from paddle import LazyGuard
            >>> from paddle.nn import Linear

            >>> with LazyGuard():
            ...     # w and b are initialized lazily and have no memory.
            ...     net = Linear(10, 10)
            ...
            >>> for param in net.parameters():
            ...     # Initialize param and allocate memory explicitly.
            ...     param.initialize()
    """

    def __enter__(self):
        if False:
            return 10
        '\n        Construct instance from class_obj by Lazy Initializing parameters.\n\n        Examples:\n\n            .. code-block:: python\n\n                >>> from paddle import LazyGuard\n                >>> from paddle.nn import Linear\n\n                >>> with LazyGuard():\n                ...     fc = LazyInit(Linear)(10, 10)\n                ...\n                >>> for param in fc.parameters():\n                ...     param.initialize()\n        '
        lazy_init_helper().enable()

    def __exit__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        lazy_init_helper().disable()