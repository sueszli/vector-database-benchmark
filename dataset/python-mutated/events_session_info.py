from msrest.serialization import Model

class EventsSessionInfo(Model):
    """Session info for an event result.

    :param id: ID of the session
    :type id: str
    """
    _attribute_map = {'id': {'key': 'id', 'type': 'str'}}

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(EventsSessionInfo, self).__init__(**kwargs)
        self.id = kwargs.get('id', None)