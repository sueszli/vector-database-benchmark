from msrest.serialization import Model

class ErrorRecord(Model):
    """ErrorRecord.

    :param id: Input document unique identifier the error refers to.
    :type id: str
    :param message: Error message.
    :type message: str
    """
    _attribute_map = {'id': {'key': 'id', 'type': 'str'}, 'message': {'key': 'message', 'type': 'str'}}

    def __init__(self, *, id: str=None, message: str=None, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(ErrorRecord, self).__init__(**kwargs)
        self.id = id
        self.message = message