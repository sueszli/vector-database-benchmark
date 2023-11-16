from msrest.serialization import Model

class AlterationsDTO(Model):
    """Collection of words that are synonyms.

    All required parameters must be populated in order to send to Azure.

    :param alterations: Required. Words that are synonymous with each other.
    :type alterations: list[str]
    """
    _validation = {'alterations': {'required': True}}
    _attribute_map = {'alterations': {'key': 'alterations', 'type': '[str]'}}

    def __init__(self, *, alterations, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super(AlterationsDTO, self).__init__(**kwargs)
        self.alterations = alterations