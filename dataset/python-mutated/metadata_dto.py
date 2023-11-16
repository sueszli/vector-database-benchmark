from msrest.serialization import Model

class MetadataDTO(Model):
    """Name - value pair of metadata.

    All required parameters must be populated in order to send to Azure.

    :param name: Required. Metadata name.
    :type name: str
    :param value: Required. Metadata value.
    :type value: str
    """
    _validation = {'name': {'required': True, 'max_length': 100, 'min_length': 1}, 'value': {'required': True, 'max_length': 500, 'min_length': 1}}
    _attribute_map = {'name': {'key': 'name', 'type': 'str'}, 'value': {'key': 'value', 'type': 'str'}}

    def __init__(self, **kwargs):
        if False:
            return 10
        super(MetadataDTO, self).__init__(**kwargs)
        self.name = kwargs.get('name', None)
        self.value = kwargs.get('value', None)