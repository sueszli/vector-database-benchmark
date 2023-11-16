from msrest.serialization import Model

class RankedAction(Model):
    """A ranked action with its resulting probability.

    Variables are only populated by the server, and will be ignored when
    sending a request.

    :ivar id: Id of the action
    :vartype id: str
    :ivar probability: Probability of the action
    :vartype probability: float
    """
    _validation = {'id': {'readonly': True, 'max_length': 256}, 'probability': {'readonly': True, 'maximum': 1, 'minimum': 0}}
    _attribute_map = {'id': {'key': 'id', 'type': 'str'}, 'probability': {'key': 'probability', 'type': 'float'}}

    def __init__(self, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(RankedAction, self).__init__(**kwargs)
        self.id = None
        self.probability = None