from msrest.serialization import Model

class LanguageBatchInput(Model):
    """LanguageBatchInput.

    :param documents:
    :type documents:
     list[~azure.cognitiveservices.language.textanalytics.models.LanguageInput]
    """
    _attribute_map = {'documents': {'key': 'documents', 'type': '[LanguageInput]'}}

    def __init__(self, **kwargs):
        if False:
            return 10
        super(LanguageBatchInput, self).__init__(**kwargs)
        self.documents = kwargs.get('documents', None)