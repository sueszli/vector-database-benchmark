from msrest.serialization import Model

class RelatedSearchesModule(Model):
    """Defines a list of related searches.

    Variables are only populated by the server, and will be ignored when
    sending a request.

    :ivar value: A list of related searches.
    :vartype value:
     list[~azure.cognitiveservices.search.visualsearch.models.Query]
    """
    _validation = {'value': {'readonly': True}}
    _attribute_map = {'value': {'key': 'value', 'type': '[Query]'}}

    def __init__(self, **kwargs) -> None:
        if False:
            print('Hello World!')
        super(RelatedSearchesModule, self).__init__(**kwargs)
        self.value = None