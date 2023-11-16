from msrest.serialization import Model

class DatabaseAccountRegenerateKeyParameters(Model):
    """Parameters to regenerate the keys within the database account.

    :param key_kind: The access key to regenerate. Possible values include:
     'primary', 'secondary', 'primaryReadonly', 'secondaryReadonly'
    :type key_kind: str or :class:`KeyKind
     <azure.mgmt.documentdb.models.KeyKind>`
    """
    _validation = {'key_kind': {'required': True}}
    _attribute_map = {'key_kind': {'key': 'keyKind', 'type': 'str'}}

    def __init__(self, key_kind):
        if False:
            print('Hello World!')
        self.key_kind = key_kind