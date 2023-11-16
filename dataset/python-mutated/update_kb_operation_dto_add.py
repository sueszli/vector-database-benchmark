from .create_kb_input_dto import CreateKbInputDTO

class UpdateKbOperationDTOAdd(CreateKbInputDTO):
    """An instance of CreateKbInputDTO for add operation.

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

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super(UpdateKbOperationDTOAdd, self).__init__(**kwargs)