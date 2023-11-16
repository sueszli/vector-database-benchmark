from msrest.serialization import Model

class LanguageBatchResultItem(Model):
    """LanguageBatchResultItem.

    :param id: Unique, non-empty document identifier.
    :type id: str
    :param detected_languages: A list of extracted languages.
    :type detected_languages:
     list[~azure.cognitiveservices.language.textanalytics.models.DetectedLanguage]
    :param statistics: (Optional) if showStats=true was specified in the
     request this field will contain information about the document payload.
    :type statistics:
     ~azure.cognitiveservices.language.textanalytics.models.DocumentStatistics
    """
    _attribute_map = {'id': {'key': 'id', 'type': 'str'}, 'detected_languages': {'key': 'detectedLanguages', 'type': '[DetectedLanguage]'}, 'statistics': {'key': 'statistics', 'type': 'DocumentStatistics'}}

    def __init__(self, *, id: str=None, detected_languages=None, statistics=None, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super(LanguageBatchResultItem, self).__init__(**kwargs)
        self.id = id
        self.detected_languages = detected_languages
        self.statistics = statistics