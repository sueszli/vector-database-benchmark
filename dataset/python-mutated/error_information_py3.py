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

    def __init__(self, *, code: str=None, inner_error=None, message: str=None, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super(ErrorInformation, self).__init__(**kwargs)
        self.code = code
        self.inner_error = inner_error
        self.message = message