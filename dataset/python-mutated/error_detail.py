from msrest.serialization import Model

class ErrorDetail(Model):
    """Error details.

    All required parameters must be populated in order to send to Azure.

    :param code: Required. The error's code.
    :type code: str
    :param message: Required. A human readable error message.
    :type message: str
    :param target: Indicates which property in the request is responsible for
     the error.
    :type target: str
    :param value: Indicates which value in 'target' is responsible for the
     error.
    :type value: str
    :param resources: Indicates resources which were responsible for the
     error.
    :type resources: list[str]
    :param additional_properties:
    :type additional_properties: object
    """
    _validation = {'code': {'required': True}, 'message': {'required': True}}
    _attribute_map = {'code': {'key': 'code', 'type': 'str'}, 'message': {'key': 'message', 'type': 'str'}, 'target': {'key': 'target', 'type': 'str'}, 'value': {'key': 'value', 'type': 'str'}, 'resources': {'key': 'resources', 'type': '[str]'}, 'additional_properties': {'key': 'additionalProperties', 'type': 'object'}}

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super(ErrorDetail, self).__init__(**kwargs)
        self.code = kwargs.get('code', None)
        self.message = kwargs.get('message', None)
        self.target = kwargs.get('target', None)
        self.value = kwargs.get('value', None)
        self.resources = kwargs.get('resources', None)
        self.additional_properties = kwargs.get('additional_properties', None)