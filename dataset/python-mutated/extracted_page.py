from msrest.serialization import Model

class ExtractedPage(Model):
    """Extraction information of a single page in a
    with a document.

    :param number: Page number.
    :type number: int
    :param height: Height of the page (in pixels).
    :type height: int
    :param width: Width of the page (in pixels).
    :type width: int
    :param cluster_id: Cluster identifier.
    :type cluster_id: int
    :param key_value_pairs: List of Key-Value pairs extracted from the page.
    :type key_value_pairs:
     list[~azure.cognitiveservices.formrecognizer.models.ExtractedKeyValuePair]
    :param tables: List of Tables and their information extracted from the
     page.
    :type tables:
     list[~azure.cognitiveservices.formrecognizer.models.ExtractedTable]
    """
    _attribute_map = {'number': {'key': 'number', 'type': 'int'}, 'height': {'key': 'height', 'type': 'int'}, 'width': {'key': 'width', 'type': 'int'}, 'cluster_id': {'key': 'clusterId', 'type': 'int'}, 'key_value_pairs': {'key': 'keyValuePairs', 'type': '[ExtractedKeyValuePair]'}, 'tables': {'key': 'tables', 'type': '[ExtractedTable]'}}

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super(ExtractedPage, self).__init__(**kwargs)
        self.number = kwargs.get('number', None)
        self.height = kwargs.get('height', None)
        self.width = kwargs.get('width', None)
        self.cluster_id = kwargs.get('cluster_id', None)
        self.key_value_pairs = kwargs.get('key_value_pairs', None)
        self.tables = kwargs.get('tables', None)