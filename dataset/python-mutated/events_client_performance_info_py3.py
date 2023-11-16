from msrest.serialization import Model

class EventsClientPerformanceInfo(Model):
    """Client performance information.

    :param name: The name of the client performance
    :type name: str
    """
    _attribute_map = {'name': {'key': 'name', 'type': 'str'}}

    def __init__(self, *, name: str=None, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super(EventsClientPerformanceInfo, self).__init__(**kwargs)
        self.name = name