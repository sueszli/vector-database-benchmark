from .delete_kb_contents_dto_py3 import DeleteKbContentsDTO

class UpdateKbOperationDTODelete(DeleteKbContentsDTO):
    """An instance of DeleteKbContentsDTO for delete Operation.

    :param ids: List of Qna Ids to be deleted
    :type ids: list[int]
    :param sources: List of sources to be deleted from knowledgebase.
    :type sources: list[str]
    """
    _attribute_map = {'ids': {'key': 'ids', 'type': '[int]'}, 'sources': {'key': 'sources', 'type': '[str]'}}

    def __init__(self, *, ids=None, sources=None, **kwargs) -> None:
        if False:
            return 10
        super(UpdateKbOperationDTODelete, self).__init__(ids=ids, sources=sources, **kwargs)