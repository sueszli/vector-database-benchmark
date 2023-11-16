from msrest.serialization import Model

class InnerError(Model):
    """InnerError.

    :param request_id:
    :type request_id: str
    """
    _attribute_map = {'request_id': {'key': 'requestId', 'type': 'str'}}

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super(InnerError, self).__init__(**kwargs)
        self.request_id = kwargs.get('request_id', None)