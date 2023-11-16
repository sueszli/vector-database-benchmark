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

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super(LanguageBatchResultItem, self).__init__(**kwargs)
        self.id = kwargs.get('id', None)
        self.detected_languages = kwargs.get('detected_languages', None)
        self.statistics = kwargs.get('statistics', None)