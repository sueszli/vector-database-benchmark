from msrest.serialization import Model

class KnowledgeRequest(Model):
    """A JSON object containing information about the request, such as filters for
    the resulting actions.

    :param filters: A key-value object consisting of filters that may be
     specified to limit the results returned by the API.
    :type filters: ~azure.cognitiveservices.search.visualsearch.models.Filters
    """
    _attribute_map = {'filters': {'key': 'filters', 'type': 'Filters'}}

    def __init__(self, *, filters=None, **kwargs) -> None:
        if False:
            print('Hello World!')
        super(KnowledgeRequest, self).__init__(**kwargs)
        self.filters = filters