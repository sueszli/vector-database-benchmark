from msrest.serialization import Model

class MultiLanguageBatchInput(Model):
    """MultiLanguageBatchInput.

    :param documents:
    :type documents:
     list[~azure.cognitiveservices.language.textanalytics.models.MultiLanguageInput]
    """
    _attribute_map = {'documents': {'key': 'documents', 'type': '[MultiLanguageInput]'}}

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(MultiLanguageBatchInput, self).__init__(**kwargs)
        self.documents = kwargs.get('documents', None)