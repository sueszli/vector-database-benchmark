from msrest.serialization import Model

class DatabaseAccountPatchParameters(Model):
    """Parameters for patching Azure DocumentDB database account properties.

    :param tags:
    :type tags: dict
    """
    _validation = {'tags': {'required': True}}
    _attribute_map = {'tags': {'key': 'tags', 'type': '{str}'}}

    def __init__(self, tags):
        if False:
            i = 10
            return i + 15
        self.tags = tags