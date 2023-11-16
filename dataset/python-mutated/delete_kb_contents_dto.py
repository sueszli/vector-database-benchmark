from msrest.serialization import Model

class DeleteKbContentsDTO(Model):
    """PATCH body schema of Delete Operation in UpdateKb.

    :param ids: List of Qna Ids to be deleted
    :type ids: list[int]
    :param sources: List of sources to be deleted from knowledgebase.
    :type sources: list[str]
    """
    _attribute_map = {'ids': {'key': 'ids', 'type': '[int]'}, 'sources': {'key': 'sources', 'type': '[str]'}}

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(DeleteKbContentsDTO, self).__init__(**kwargs)
        self.ids = kwargs.get('ids', None)
        self.sources = kwargs.get('sources', None)