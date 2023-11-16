from msrest.serialization import Model

class KeysResult(Model):
    """Result of an operation to get
    the keys extracted by a model.

    :param clusters: Object mapping ClusterIds to Key lists.
    :type clusters: dict[str, list[str]]
    """
    _attribute_map = {'clusters': {'key': 'clusters', 'type': '{[str]}'}}

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(KeysResult, self).__init__(**kwargs)
        self.clusters = kwargs.get('clusters', None)