from msrest.serialization import Model

class EventsClientPerformanceInfo(Model):
    """Client performance information.

    :param name: The name of the client performance
    :type name: str
    """
    _attribute_map = {'name': {'key': 'name', 'type': 'str'}}

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(EventsClientPerformanceInfo, self).__init__(**kwargs)
        self.name = kwargs.get('name', None)