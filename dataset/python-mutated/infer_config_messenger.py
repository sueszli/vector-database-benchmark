from .messenger import Messenger

class InferConfigMessenger(Messenger):
    """
    Given a callable `fn` that contains Pyro primitive calls
    and a callable `config_fn` taking a trace site and returning a dictionary,
    updates the value of the infer kwarg at a sample site to config_fn(site).

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param config_fn: a callable taking a site and returning an infer dict
    :returns: stochastic function decorated with :class:`~pyro.poutine.infer_config_messenger.InferConfigMessenger`
    """

    def __init__(self, config_fn):
        if False:
            while True:
                i = 10
        "\n        :param config_fn: a callable taking a site and returning an infer dict\n\n        Constructor. Doesn't do much, just stores the stochastic function\n        and the config_fn.\n        "
        super().__init__()
        self.config_fn = config_fn

    def _pyro_sample(self, msg):
        if False:
            i = 10
            return i + 15
        '\n        :param msg: current message at a trace site.\n\n        If self.config_fn is not None, calls self.config_fn on msg\n        and stores the result in msg["infer"].\n\n        Otherwise, implements default sampling behavior\n        with no additional effects.\n        '
        msg['infer'].update(self.config_fn(msg))
        return None

    def _pyro_param(self, msg):
        if False:
            print('Hello World!')
        '\n        :param msg: current message at a trace site.\n\n        If self.config_fn is not None, calls self.config_fn on msg\n        and stores the result in msg["infer"].\n\n        Otherwise, implements default param behavior\n        with no additional effects.\n        '
        msg['infer'].update(self.config_fn(msg))
        return None