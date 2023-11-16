from msrest.serialization import Model

class ExtractedKeyValuePair(Model):
    """Representation of a key-value pair as a list
    of key and value tokens.

    :param key: List of tokens for the extracted key in a key-value pair.
    :type key:
     list[~azure.cognitiveservices.formrecognizer.models.ExtractedToken]
    :param value: List of tokens for the extracted value in a key-value pair.
    :type value:
     list[~azure.cognitiveservices.formrecognizer.models.ExtractedToken]
    """
    _attribute_map = {'key': {'key': 'key', 'type': '[ExtractedToken]'}, 'value': {'key': 'value', 'type': '[ExtractedToken]'}}

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(ExtractedKeyValuePair, self).__init__(**kwargs)
        self.key = kwargs.get('key', None)
        self.value = kwargs.get('value', None)