from .response_base import ResponseBase

class Identifiable(ResponseBase):
    """Defines the identity of a resource.

    You probably want to use the sub-classes and not this class directly. Known
    sub-classes are: Response

    Variables are only populated by the server, and will be ignored when
    sending a request.

    All required parameters must be populated in order to send to Azure.

    :param _type: Required. Constant filled by server.
    :type _type: str
    :ivar id: A String identifier.
    :vartype id: str
    """
    _validation = {'_type': {'required': True}, 'id': {'readonly': True}}
    _attribute_map = {'_type': {'key': '_type', 'type': 'str'}, 'id': {'key': 'id', 'type': 'str'}}
    _subtype_map = {'_type': {'Response': 'Response'}}

    def __init__(self, **kwargs):
        if False:
            return 10
        super(Identifiable, self).__init__(**kwargs)
        self.id = None
        self._type = 'Identifiable'