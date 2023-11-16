"""Support for wrapping converted functions bodies with auxiliary logic."""
from nvidia.dali._autograph.core import ag_ctx
from nvidia.dali._autograph.core import converter
from nvidia.dali._autograph.operators import variables

class FunctionScope(object):
    """Context manager that wraps the body of a converted function.

  This context manager handles various operations related to the scope of a
  function:
    * optional TF name scopes - these name scopes match the name of the
        function, for easy visualization in tensorBoard;
    * optional automatic control dependencies - this adds the same mechanism
        for control dependencies that is used by `@tf.function`; it can be
        optionally enabled when using `tf.autograph.to_graph`;
    * tracking of autograph conversion state (whether it's enabled by the user,
        conversion options);
  """

    def __init__(self, function_name, scope_name, options):
        if False:
            for i in range(10):
                print('nop')
        self.name = scope_name
        self.options = options
        if options.user_requested:
            self.autograph_ctx = ag_ctx.ControlStatusCtx(ag_ctx.Status.ENABLED, options)
        self.callopts = options.call_options()

    def _sanitize(self, name):
        if False:
            for i in range(10):
                print('nop')
        'See https://www.tensorflow.org/api_docs/python/tf/Graph#name_scope.'
        if name and name.startswith('_'):
            name = 'fn' + name
        return name

    def __enter__(self):
        if False:
            return 10
        if self.options.user_requested:
            self.autograph_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            return 10
        if self.options.user_requested:
            self.autograph_ctx.__exit__(exc_type, exc_val, exc_tb)

    def ret(self, value, did_return):
        if False:
            return 10
        'Marks a value as returned from the function guarded by the scope.'
        del did_return
        if isinstance(value, variables.UndefinedReturnValue):
            return None
        return value

def with_function_scope(thunk, scope_name, options):
    if False:
        print('Hello World!')
    'Inline version of the FunctionScope context manager.'
    with FunctionScope('lambda_', scope_name, options) as scope:
        return thunk(scope)