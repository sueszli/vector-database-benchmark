from msrest.serialization import Model
from msrest.exceptions import HttpOperationError

class ErrorResponse(Model):
    """Error details.

    Contains details when the response code indicates an error.

    All required parameters must be populated in order to send to Azure.

    :param error: Required. The error details.
    :type error: ~azure.applicationinsights.models.ErrorInfo
    """
    _validation = {'error': {'required': True}}
    _attribute_map = {'error': {'key': 'error', 'type': 'ErrorInfo'}}

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(ErrorResponse, self).__init__(**kwargs)
        self.error = kwargs.get('error', None)

class ErrorResponseException(HttpOperationError):
    """Server responded with exception of type: 'ErrorResponse'.

    :param deserialize: A deserializer
    :param response: Server response to be deserialized.
    """

    def __init__(self, deserialize, response, *args):
        if False:
            for i in range(10):
                print('nop')
        super(ErrorResponseException, self).__init__(deserialize, response, 'ErrorResponse', *args)