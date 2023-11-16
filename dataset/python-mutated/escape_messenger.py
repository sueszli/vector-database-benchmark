from .messenger import Messenger
from .runtime import NonlocalExit

class EscapeMessenger(Messenger):
    """
    Messenger that does a nonlocal exit by raising a util.NonlocalExit exception
    """

    def __init__(self, escape_fn):
        if False:
            print('Hello World!')
        '\n        :param escape_fn: function that takes a msg as input and returns True\n            if the poutine should perform a nonlocal exit at that site.\n\n        Constructor.  Stores fn and escape_fn.\n        '
        super().__init__()
        self.escape_fn = escape_fn

    def _pyro_sample(self, msg):
        if False:
            while True:
                i = 10
        '\n        :param msg: current message at a trace site\n        :returns: a sample from the stochastic function at the site.\n\n        Evaluates self.escape_fn on the site (self.escape_fn(msg)).\n\n        If this returns True, raises an exception NonlocalExit(msg).\n        Else, implements default _pyro_sample behavior with no additional effects.\n        '
        if self.escape_fn(msg):
            msg['done'] = True
            msg['stop'] = True

            def cont(m):
                if False:
                    print('Hello World!')
                raise NonlocalExit(m)
            msg['continuation'] = cont
        return None