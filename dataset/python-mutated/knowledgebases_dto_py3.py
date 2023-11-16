from msrest.serialization import Model

class KnowledgebasesDTO(Model):
    """Collection of knowledgebases owned by a user.

    :param knowledgebases: Collection of knowledgebase records.
    :type knowledgebases:
     list[~azure.cognitiveservices.knowledge.qnamaker.models.KnowledgebaseDTO]
    """
    _attribute_map = {'knowledgebases': {'key': 'knowledgebases', 'type': '[KnowledgebaseDTO]'}}

    def __init__(self, *, knowledgebases=None, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super(KnowledgebasesDTO, self).__init__(**kwargs)
        self.knowledgebases = knowledgebases