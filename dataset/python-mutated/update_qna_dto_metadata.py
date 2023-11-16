from .update_metadata_dto import UpdateMetadataDTO

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

    def __init__(self, **kwargs):
        if False:
            return 10
        super(UpdateQnaDTOMetadata, self).__init__(**kwargs)