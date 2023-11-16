from msrest.serialization import Model

class FormOperationError(Model):
    """Error reported during an operation.

    :param error_message: Message reported during the train operation.
    :type error_message: str
    """
    _attribute_map = {'error_message': {'key': 'errorMessage', 'type': 'str'}}

    def __init__(self, *, error_message: str=None, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super(FormOperationError, self).__init__(**kwargs)
        self.error_message = error_message