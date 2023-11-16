from .util import to_unicode

class CLIArgs(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._top_kwargs = {}
        self._step_kwargs = {}

    def _set_step_kwargs(self, kwargs):
        if False:
            print('Hello World!')
        self._step_kwargs = kwargs

    def _set_top_kwargs(self, kwargs):
        if False:
            while True:
                i = 10
        self._top_kwargs = kwargs

    @property
    def top_kwargs(self):
        if False:
            return 10
        return self._top_kwargs

    @property
    def step_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        return self._step_kwargs

    def step_command(self, executable, script, step_name, top_kwargs=None, step_kwargs=None):
        if False:
            i = 10
            return i + 15
        cmd = [executable, '-u', script]
        if top_kwargs is None:
            top_kwargs = self._top_kwargs
        if step_kwargs is None:
            step_kwargs = self._step_kwargs
        top_args_list = list(self._options(top_kwargs))
        cmd.extend(top_args_list)
        cmd.extend(['step', step_name])
        step_args_list = list(self._options(step_kwargs))
        cmd.extend(step_args_list)
        return cmd

    @staticmethod
    def _options(mapping):
        if False:
            i = 10
            return i + 15
        for (k, v) in mapping.items():
            if v is None or v is False:
                continue
            if k == 'decospecs':
                k = 'with'
            k = k.replace('_', '-')
            v = v if isinstance(v, (list, tuple, set)) else [v]
            for value in v:
                yield ('--%s' % k)
                if not isinstance(value, bool):
                    yield to_unicode(value)
cli_args = CLIArgs()