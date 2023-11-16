from msrest.serialization import Model

class KeyPhraseBatchResultItem(Model):
    """KeyPhraseBatchResultItem.

    Variables are only populated by the server, and will be ignored when
    sending a request.

    :param id: Unique, non-empty document identifier.
    :type id: str
    :ivar key_phrases: A list of representative words or phrases. The number
     of key phrases returned is proportional to the number of words in the
     input document.
    :vartype key_phrases: list[str]
    :param statistics: (Optional) if showStats=true was specified in the
     request this field will contain information about the document payload.
    :type statistics:
     ~azure.cognitiveservices.language.textanalytics.models.DocumentStatistics
    """
    _validation = {'key_phrases': {'readonly': True}}
    _attribute_map = {'id': {'key': 'id', 'type': 'str'}, 'key_phrases': {'key': 'keyPhrases', 'type': '[str]'}, 'statistics': {'key': 'statistics', 'type': 'DocumentStatistics'}}

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super(KeyPhraseBatchResultItem, self).__init__(**kwargs)
        self.id = kwargs.get('id', None)
        self.key_phrases = None
        self.statistics = kwargs.get('statistics', None)