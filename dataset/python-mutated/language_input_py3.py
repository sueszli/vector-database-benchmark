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

    def __init__(self, *, country_hint: str=None, id: str=None, text: str=None, **kwargs) -> None:
        if False:
            return 10
        super(LanguageInput, self).__init__(**kwargs)
        self.country_hint = country_hint
        self.id = id
        self.text = text