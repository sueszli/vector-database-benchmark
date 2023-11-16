from msrest.serialization import Model

class ErrorInfo(Model):
    """The code and message for an error.

    All required parameters must be populated in order to send to Azure.

    :param code: Required. A machine readable error code.
    :type code: str
    :param message: Required. A human readable error message.
    :type message: str
    :param details: error details.
    :type details: list[~azure.applicationinsights.models.ErrorDetail]
    :param innererror: Inner error details if they exist.
    :type innererror: ~azure.applicationinsights.models.ErrorInfo
    :param additional_properties:
    :type additional_properties: object
    """
    _validation = {'code': {'required': True}, 'message': {'required': True}}
    _attribute_map = {'code': {'key': 'code', 'type': 'str'}, 'message': {'key': 'message', 'type': 'str'}, 'details': {'key': 'details', 'type': '[ErrorDetail]'}, 'innererror': {'key': 'innererror', 'type': 'ErrorInfo'}, 'additional_properties': {'key': 'additionalProperties', 'type': 'object'}}

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(ErrorInfo, self).__init__(**kwargs)
        self.code = kwargs.get('code', None)
        self.message = kwargs.get('message', None)
        self.details = kwargs.get('details', None)
        self.innererror = kwargs.get('innererror', None)
        self.additional_properties = kwargs.get('additional_properties', None)