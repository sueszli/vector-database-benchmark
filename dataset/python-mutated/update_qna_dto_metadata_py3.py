from .update_metadata_dto_py3 import UpdateMetadataDTO

class UpdateQnaDTOMetadata(UpdateMetadataDTO):
    """List of metadata associated with the answer to be updated.

    :param delete: List of Metadata associated with answer to be deleted
    :type delete:
     list[~azure.cognitiveservices.knowledge.qnamaker.models.MetadataDTO]
    :param add: List of metadata associated with answer to be added
    :type add:
     list[~azure.cognitiveservices.knowledge.qnamaker.models.MetadataDTO]
    """
    _attribute_map = {'delete': {'key': 'delete', 'type': '[MetadataDTO]'}, 'add': {'key': 'add', 'type': '[MetadataDTO]'}}

    def __init__(self, *, delete=None, add=None, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super(UpdateQnaDTOMetadata, self).__init__(delete=delete, add=add, **kwargs)