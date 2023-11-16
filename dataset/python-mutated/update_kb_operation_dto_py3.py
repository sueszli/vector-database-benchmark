from msrest.serialization import Model

class UpdateKbOperationDTO(Model):
    """Contains list of QnAs to be updated.

    :param add: An instance of CreateKbInputDTO for add operation
    :type add:
     ~azure.cognitiveservices.knowledge.qnamaker.models.UpdateKbOperationDTOAdd
    :param delete: An instance of DeleteKbContentsDTO for delete Operation
    :type delete:
     ~azure.cognitiveservices.knowledge.qnamaker.models.UpdateKbOperationDTODelete
    :param update: An instance of UpdateKbContentsDTO for Update Operation
    :type update:
     ~azure.cognitiveservices.knowledge.qnamaker.models.UpdateKbOperationDTOUpdate
    :param enable_hierarchical_extraction: Enable hierarchical extraction of
     Q-A from files and urls. The value set during KB creation will be used if
     this field is not present.
    :type enable_hierarchical_extraction: bool
    :param default_answer_used_for_extraction: Text string to be used as the
     answer in any Q-A which has no extracted answer from the document but has
     a hierarchy. Required when EnableHierarchicalExtraction field is set to
     True.
    :type default_answer_used_for_extraction: str
    """
    _validation = {'default_answer_used_for_extraction': {'max_length': 300, 'min_length': 1}}
    _attribute_map = {'add': {'key': 'add', 'type': 'UpdateKbOperationDTOAdd'}, 'delete': {'key': 'delete', 'type': 'UpdateKbOperationDTODelete'}, 'update': {'key': 'update', 'type': 'UpdateKbOperationDTOUpdate'}, 'enable_hierarchical_extraction': {'key': 'enableHierarchicalExtraction', 'type': 'bool'}, 'default_answer_used_for_extraction': {'key': 'defaultAnswerUsedForExtraction', 'type': 'str'}}

    def __init__(self, *, add=None, delete=None, update=None, enable_hierarchical_extraction: bool=None, default_answer_used_for_extraction: str=None, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(UpdateKbOperationDTO, self).__init__(**kwargs)
        self.add = add
        self.delete = delete
        self.update = update
        self.enable_hierarchical_extraction = enable_hierarchical_extraction
        self.default_answer_used_for_extraction = default_answer_used_for_extraction