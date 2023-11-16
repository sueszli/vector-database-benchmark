from .messenger import Messenger

class UnconditionMessenger(Messenger):
    """
    Messenger to force the value of observed nodes to be sampled from their
    distribution, ignoring observations.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()

    def _pyro_sample(self, msg):
        if False:
            return 10
        '\n        :param msg: current message at a trace site.\n\n        Samples value from distribution, irrespective of whether or not the\n        node has an observed value.\n        '
        if msg['is_observed']:
            msg['is_observed'] = False
            msg['infer']['was_observed'] = True
            msg['infer']['obs'] = msg['value']
            msg['value'] = None
            msg['done'] = False
        return None