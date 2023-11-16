from msrest.serialization import Model

class DeleteKbContentsDTO(Model):
    """PATCH body schema of Delete Operation in UpdateKb.

    :param ids: List of Qna Ids to be deleted
    :type ids: list[int]
    :param sources: List of sources to be deleted from knowledgebase.
    :type sources: list[str]
    """
    _attribute_map = {'ids': {'key': 'ids', 'type': '[int]'}, 'sources': {'key': 'sources', 'type': '[str]'}}

    def __init__(self, *, ids=None, sources=None, **kwargs) -> None:
        if False:
            return 10
        super(DeleteKbContentsDTO, self).__init__(**kwargs)
        self.ids = ids
        self.sources = sources