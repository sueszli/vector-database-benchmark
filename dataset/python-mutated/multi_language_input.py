from msrest.serialization import Model

class MultiLanguageInput(Model):
    """MultiLanguageInput.

    :param language: This is the 2 letter ISO 639-1 representation of a
     language. For example, use "en" for English; "es" for Spanish etc.,
    :type language: str
    :param id: Unique, non-empty document identifier.
    :type id: str
    :param text:
    :type text: str
    """
    _attribute_map = {'language': {'key': 'language', 'type': 'str'}, 'id': {'key': 'id', 'type': 'str'}, 'text': {'key': 'text', 'type': 'str'}}

    def __init__(self, **kwargs):
        if False:
            return 10
        super(MultiLanguageInput, self).__init__(**kwargs)
        self.language = kwargs.get('language', None)
        self.id = kwargs.get('id', None)
        self.text = kwargs.get('text', None)