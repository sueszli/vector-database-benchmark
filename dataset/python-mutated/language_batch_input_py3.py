from msrest.serialization import Model

class LanguageBatchInput(Model):
    """LanguageBatchInput.

    :param documents:
    :type documents:
     list[~azure.cognitiveservices.language.textanalytics.models.LanguageInput]
    """
    _attribute_map = {'documents': {'key': 'documents', 'type': '[LanguageInput]'}}

    def __init__(self, *, documents=None, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(LanguageBatchInput, self).__init__(**kwargs)
        self.documents = documents