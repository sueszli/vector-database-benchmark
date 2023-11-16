from msrest.serialization import Model

class EventsOperationInfo(Model):
    """Operation info for an event result.

    :param name: Name of the operation
    :type name: str
    :param id: ID of the operation
    :type id: str
    :param parent_id: Parent ID of the operation
    :type parent_id: str
    :param synthetic_source: Synthetic source of the operation
    :type synthetic_source: str
    """
    _attribute_map = {'name': {'key': 'name', 'type': 'str'}, 'id': {'key': 'id', 'type': 'str'}, 'parent_id': {'key': 'parentId', 'type': 'str'}, 'synthetic_source': {'key': 'syntheticSource', 'type': 'str'}}

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(EventsOperationInfo, self).__init__(**kwargs)
        self.name = kwargs.get('name', None)
        self.id = kwargs.get('id', None)
        self.parent_id = kwargs.get('parent_id', None)
        self.synthetic_source = kwargs.get('synthetic_source', None)