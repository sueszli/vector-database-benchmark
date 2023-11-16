from msrest.serialization import Model

class KnowledgebasesDTO(Model):
    """Collection of knowledgebases owned by a user.

    :param knowledgebases: Collection of knowledgebase records.
    :type knowledgebases:
     list[~azure.cognitiveservices.knowledge.qnamaker.models.KnowledgebaseDTO]
    """
    _attribute_map = {'knowledgebases': {'key': 'knowledgebases', 'type': '[KnowledgebaseDTO]'}}

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(KnowledgebasesDTO, self).__init__(**kwargs)
        self.knowledgebases = kwargs.get('knowledgebases', None)