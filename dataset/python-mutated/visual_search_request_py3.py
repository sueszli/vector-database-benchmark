from msrest.serialization import Model

class VisualSearchRequest(Model):
    """A JSON object that contains information about the image to get insights of.
    Specify this object only in a knowledgeRequest form data.

    :param image_info: A JSON object that identities the image to get insights
     of.
    :type image_info:
     ~azure.cognitiveservices.search.visualsearch.models.ImageInfo
    :param knowledge_request: A JSON object containing information about the
     request, such as filters, or a description.
    :type knowledge_request:
     ~azure.cognitiveservices.search.visualsearch.models.KnowledgeRequest
    """
    _attribute_map = {'image_info': {'key': 'imageInfo', 'type': 'ImageInfo'}, 'knowledge_request': {'key': 'knowledgeRequest', 'type': 'KnowledgeRequest'}}

    def __init__(self, *, image_info=None, knowledge_request=None, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super(VisualSearchRequest, self).__init__(**kwargs)
        self.image_info = image_info
        self.knowledge_request = knowledge_request