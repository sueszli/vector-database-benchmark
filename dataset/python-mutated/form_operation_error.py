from msrest.serialization import Model

class FormOperationError(Model):
    """Error reported during an operation.

    :param error_message: Message reported during the train operation.
    :type error_message: str
    """
    _attribute_map = {'error_message': {'key': 'errorMessage', 'type': 'str'}}

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super(FormOperationError, self).__init__(**kwargs)
        self.error_message = kwargs.get('error_message', None)