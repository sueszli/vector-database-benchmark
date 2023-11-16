from .identifiable_py3 import Identifiable

class Response(Identifiable):
    """Defines a response. All schemas that return at the root of the response
    must inherit from this object.

    You probably want to use the sub-classes and not this class directly. Known
    sub-classes are: ImageKnowledge, ErrorResponse, Thing

    Variables are only populated by the server, and will be ignored when
    sending a request.

    All required parameters must be populated in order to send to Azure.

    :param _type: Required. Constant filled by server.
    :type _type: str
    :ivar id: A String identifier.
    :vartype id: str
    :ivar read_link: The URL that returns this resource. To use the URL,
     append query parameters as appropriate and include the
     Ocp-Apim-Subscription-Key header.
    :vartype read_link: str
    :ivar web_search_url: The URL to Bing's search result for this item.
    :vartype web_search_url: str
    """
    _validation = {'_type': {'required': True}, 'id': {'readonly': True}, 'read_link': {'readonly': True}, 'web_search_url': {'readonly': True}}
    _attribute_map = {'_type': {'key': '_type', 'type': 'str'}, 'id': {'key': 'id', 'type': 'str'}, 'read_link': {'key': 'readLink', 'type': 'str'}, 'web_search_url': {'key': 'webSearchUrl', 'type': 'str'}}
    _subtype_map = {'_type': {'ImageKnowledge': 'ImageKnowledge', 'ErrorResponse': 'ErrorResponse', 'Thing': 'Thing'}}

    def __init__(self, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super(Response, self).__init__(**kwargs)
        self.read_link = None
        self.web_search_url = None
        self._type = 'Response'