from msrest.serialization import Model

class TrainRequest(Model):
    """Contract to initiate a train request.

    All required parameters must be populated in order to send to Azure.

    :param source: Required. Get or set source path.
    :type source: str
    """
    _validation = {'source': {'required': True, 'max_length': 2048, 'min_length': 0}}
    _attribute_map = {'source': {'key': 'source', 'type': 'str'}}

    def __init__(self, *, source: str, **kwargs) -> None:
        if False:
            return 10
        super(TrainRequest, self).__init__(**kwargs)
        self.source = source