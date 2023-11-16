from msrest.serialization import Model

class InternalError(Model):
    """InternalError.

    :param code:
    :type code: str
    :param message:
    :type message: str
    :param inner_error:
    :type inner_error:
     ~azure.cognitiveservices.language.textanalytics.models.InternalError
    """
    _attribute_map = {'code': {'key': 'code', 'type': 'str'}, 'message': {'key': 'message', 'type': 'str'}, 'inner_error': {'key': 'innerError', 'type': 'InternalError'}}

    def __init__(self, *, code: str=None, message: str=None, inner_error=None, **kwargs) -> None:
        if False:
            return 10
        super(InternalError, self).__init__(**kwargs)
        self.code = code
        self.message = message
        self.inner_error = inner_error