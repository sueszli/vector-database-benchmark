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

    def __init__(self, *, name: str=None, id: str=None, parent_id: str=None, synthetic_source: str=None, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(EventsOperationInfo, self).__init__(**kwargs)
        self.name = name
        self.id = id
        self.parent_id = parent_id
        self.synthetic_source = synthetic_source