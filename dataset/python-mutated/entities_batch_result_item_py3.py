from msrest.serialization import Model

class EntitiesBatchResultItem(Model):
    """EntitiesBatchResultItem.

    Variables are only populated by the server, and will be ignored when
    sending a request.

    :param id: Unique, non-empty document identifier.
    :type id: str
    :ivar entities: Recognized entities in the document.
    :vartype entities:
     list[~azure.cognitiveservices.language.textanalytics.models.EntityRecord]
    :param statistics: (Optional) if showStats=true was specified in the
     request this field will contain information about the document payload.
    :type statistics:
     ~azure.cognitiveservices.language.textanalytics.models.DocumentStatistics
    """
    _validation = {'entities': {'readonly': True}}
    _attribute_map = {'id': {'key': 'id', 'type': 'str'}, 'entities': {'key': 'entities', 'type': '[EntityRecord]'}, 'statistics': {'key': 'statistics', 'type': 'DocumentStatistics'}}

    def __init__(self, *, id: str=None, statistics=None, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super(EntitiesBatchResultItem, self).__init__(**kwargs)
        self.id = id
        self.entities = None
        self.statistics = statistics