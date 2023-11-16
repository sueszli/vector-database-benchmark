from msrest.serialization import Model

class LanguageInput(Model):
    """LanguageInput.

    :param country_hint:
    :type country_hint: str
    :param id: Unique, non-empty document identifier.
    :type id: str
    :param text:
    :type text: str
    """
    _attribute_map = {'country_hint': {'key': 'countryHint', 'type': 'str'}, 'id': {'key': 'id', 'type': 'str'}, 'text': {'key': 'text', 'type': 'str'}}

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(LanguageInput, self).__init__(**kwargs)
        self.country_hint = kwargs.get('country_hint', None)
        self.id = kwargs.get('id', None)
        self.text = kwargs.get('text', None)