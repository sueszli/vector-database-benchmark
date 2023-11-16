import six
from chainer import configuration
from chainer.training import extension
from chainer import variable

class unchain_variables(extension.Extension):
    """Trainer extension to unchain all comptational graphs.

    This extenstion unchains all comptational graphs after all extensions are
    run to release memory and to avoid memory leak.
    This extension can be used as a last resort when there is an extension that
    use a variable graph and cannot release the graph in itself.
    It observes the previous ``chainer.config.keep_graph_on_report`` flag.
    The extension is triggered when the flag is turned on.

    """
    priority = 0

    def __init__(self):
        if False:
            print('Hello World!')
        self._prev_flag = None

    def initialize(self, _):
        if False:
            return 10
        self._prev_flag = configuration.config.keep_graph_on_report

    def trigger(self, _):
        if False:
            for i in range(10):
                print('nop')
        flag = self._prev_flag
        self._prev_flag = configuration.config.keep_graph_on_report
        return flag

    def __call__(self, trainer):
        if False:
            i = 10
            return i + 15
        for var in six.itervalues(trainer.observation):
            if isinstance(var, variable.Variable):
                var.unchain_backward()