"""Thread-local context managers for AutoGraph."""
import enum
import inspect
import threading
from nvidia.dali._autograph.utils import ag_logging
from nvidia.dali._autograph.utils.all_utils import export_symbol
stacks = threading.local()

def _control_ctx():
    if False:
        while True:
            i = 10
    if not hasattr(stacks, 'control_status'):
        stacks.control_status = [_default_control_status_ctx()]
    return stacks.control_status

@export_symbol('__internal__.autograph.control_status_ctx', v1=[])
def control_status_ctx():
    if False:
        while True:
            i = 10
    'Returns the current control context for autograph.\n\n  This method is useful when calling `tf.__internal__.autograph.tf_convert`,\n  The context will be used by tf_convert to determine whether it should convert\n  the input function. See the sample usage like below:\n\n  ```\n  def foo(func):\n    return tf.__internal__.autograph.tf_convert(\n       input_fn, ctx=tf.__internal__.autograph.control_status_ctx())()\n  ```\n\n  Returns:\n    The current control context of autograph.\n  '
    ret = _control_ctx()[-1]
    return ret

class Status(enum.Enum):
    UNSPECIFIED = 0
    ENABLED = 1
    DISABLED = 2

class ControlStatusCtx(object):
    """A context that tracks whether autograph is enabled by the user."""

    def __init__(self, status, options=None):
        if False:
            while True:
                i = 10
        self.status = status
        self.options = options

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        _control_ctx().append(self)
        return self

    def __repr__(self):
        if False:
            return 10
        return '{}[status={}, options={}]'.format(self.__class__.__name__, self.status, self.options)

    def __exit__(self, unused_type, unused_value, unused_traceback):
        if False:
            i = 10
            return i + 15
        assert _control_ctx()[-1] is self
        _control_ctx().pop()

class NullCtx(object):
    """Helper substitute for contextlib.nullcontext."""

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        pass

    def __exit__(self, unused_type, unused_value, unused_traceback):
        if False:
            return 10
        pass

def _default_control_status_ctx():
    if False:
        print('Hello World!')
    return ControlStatusCtx(status=Status.UNSPECIFIED)
INSPECT_SOURCE_SUPPORTED = True
try:
    inspect.getsource(ag_logging.log)
except OSError:
    INSPECT_SOURCE_SUPPORTED = False
    ag_logging.warning('AutoGraph is not available in this environment: functions lack code information. This is typical of some environments like the interactive Python shell, functions with native bindings or functions created dynamically using `exec` or `eval`. Use `inspect.findsource` to check if the source code is available for the function you are trying to convert.')