from msrest.serialization import Model

class CreateKbInputDTO(Model):
    """Input to create KB.

    :param qna_list: List of QNA to be added to the index. Ids are generated
     by the service and should be omitted.
    :type qna_list:
     list[~azure.cognitiveservices.knowledge.qnamaker.models.QnADTO]
    :param urls: List of URLs to be added to knowledgebase.
    :type urls: list[str]
    :param files: List of files to be added to knowledgebase.
    :type files:
     list[~azure.cognitiveservices.knowledge.qnamaker.models.FileDTO]
    """
    _attribute_map = {'qna_list': {'key': 'qnaList', 'type': '[QnADTO]'}, 'urls': {'key': 'urls', 'type': '[str]'}, 'files': {'key': 'files', 'type': '[FileDTO]'}}

    def __init__(self, *, qna_list=None, urls=None, files=None, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(CreateKbInputDTO, self).__init__(**kwargs)
        self.qna_list = qna_list
        self.urls = urls
        self.files = files