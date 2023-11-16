from msrest.serialization import Model

class QnADocumentsDTO(Model):
    """List of QnADTO.

    :param qna_documents: List of answers.
    :type qna_documents:
     list[~azure.cognitiveservices.knowledge.qnamaker.models.QnADTO]
    """
    _attribute_map = {'qna_documents': {'key': 'qnaDocuments', 'type': '[QnADTO]'}}

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super(QnADocumentsDTO, self).__init__(**kwargs)
        self.qna_documents = kwargs.get('qna_documents', None)