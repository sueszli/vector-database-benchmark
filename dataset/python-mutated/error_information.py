from msrest.serialization import Model

class ErrorInformation(Model):
    """ErrorInformation.

    :param code:
    :type code: str
    :param inner_error:
    :type inner_error:
     ~azure.cognitiveservices.formrecognizer.models.InnerError
    :param message:
    :type message: str
    """
    _attribute_map = {'code': {'key': 'code', 'type': 'str'}, 'inner_error': {'key': 'innerError', 'type': 'InnerError'}, 'message': {'key': 'message', 'type': 'str'}}

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(ErrorInformation, self).__init__(**kwargs)
        self.code = kwargs.get('code', None)
        self.inner_error = kwargs.get('inner_error', None)
        self.message = kwargs.get('message', None)