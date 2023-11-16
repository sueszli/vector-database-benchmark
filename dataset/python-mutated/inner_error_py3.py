from msrest.serialization import Model

class InnerError(Model):
    """InnerError.

    :param request_id:
    :type request_id: str
    """
    _attribute_map = {'request_id': {'key': 'requestId', 'type': 'str'}}

    def __init__(self, *, request_id: str=None, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(InnerError, self).__init__(**kwargs)
        self.request_id = request_id