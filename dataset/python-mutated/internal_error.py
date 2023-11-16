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

    def __init__(self, **kwargs):
        if False:
            return 10
        super(InternalError, self).__init__(**kwargs)
        self.code = kwargs.get('code', None)
        self.message = kwargs.get('message', None)
        self.inner_error = kwargs.get('inner_error', None)