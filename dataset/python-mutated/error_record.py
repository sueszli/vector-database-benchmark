from msrest.serialization import Model

class ErrorRecord(Model):
    """ErrorRecord.

    :param id: Input document unique identifier the error refers to.
    :type id: str
    :param message: Error message.
    :type message: str
    """
    _attribute_map = {'id': {'key': 'id', 'type': 'str'}, 'message': {'key': 'message', 'type': 'str'}}

    def __init__(self, **kwargs):
        if False:
            return 10
        super(ErrorRecord, self).__init__(**kwargs)
        self.id = kwargs.get('id', None)
        self.message = kwargs.get('message', None)