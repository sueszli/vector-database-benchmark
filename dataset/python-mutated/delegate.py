from interface import implements
from zipline.utils.compat import ExitStack, contextmanager, wraps
from .iface import PipelineHooks, PIPELINE_HOOKS_CONTEXT_MANAGERS
from .no import NoHooks

def delegating_hooks_method(method_name):
    if False:
        print('Hello World!')
    'Factory function for making DelegatingHooks methods.\n    '
    if method_name in PIPELINE_HOOKS_CONTEXT_MANAGERS:

        @wraps(getattr(PipelineHooks, method_name))
        @contextmanager
        def ctx(self, *args, **kwargs):
            if False:
                return 10
            with ExitStack() as stack:
                for hook in self._hooks:
                    sub_ctx = getattr(hook, method_name)(*args, **kwargs)
                    stack.enter_context(sub_ctx)
                yield stack
        return ctx
    else:

        @wraps(getattr(PipelineHooks, method_name))
        def method(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            for hook in self._hooks:
                sub_method = getattr(hook, method_name)
                sub_method(*args, **kwargs)
        return method

class DelegatingHooks(implements(PipelineHooks)):
    """A PipelineHooks that delegates to one or more other hooks.

    Parameters
    ----------
    hooks : list[implements(PipelineHooks)]
        Sequence of hooks to delegate to.
    """

    def __new__(cls, hooks):
        if False:
            i = 10
            return i + 15
        if len(hooks) == 0:
            return NoHooks()
        elif len(hooks) == 1:
            return hooks[0]
        else:
            self = super(DelegatingHooks, cls).__new__(cls)
            self._hooks = hooks
            return self
    locals().update({name: delegating_hooks_method(name) for name in PipelineHooks._signatures})
del delegating_hooks_method