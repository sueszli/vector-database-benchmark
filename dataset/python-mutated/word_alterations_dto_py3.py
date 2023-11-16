from msrest.serialization import Model

class WordAlterationsDTO(Model):
    """Collection of word alterations.

    All required parameters must be populated in order to send to Azure.

    :param word_alterations: Required. Collection of word alterations.
    :type word_alterations:
     list[~azure.cognitiveservices.knowledge.qnamaker.models.AlterationsDTO]
    """
    _validation = {'word_alterations': {'required': True}}
    _attribute_map = {'word_alterations': {'key': 'wordAlterations', 'type': '[AlterationsDTO]'}}

    def __init__(self, *, word_alterations, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(WordAlterationsDTO, self).__init__(**kwargs)
        self.word_alterations = word_alterations