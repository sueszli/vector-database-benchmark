from msrest.serialization import Model

class UpdateMetadataDTO(Model):
    """PATCH Body schema to represent list of Metadata to be updated.

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
            for i in range(10):
                print('nop')
        super(UpdateMetadataDTO, self).__init__(**kwargs)
        self.delete = kwargs.get('delete', None)
        self.add = kwargs.get('add', None)