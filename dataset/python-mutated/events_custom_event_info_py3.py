from msrest.serialization import Model

class EventsCustomEventInfo(Model):
    """The custom event information.

    :param name: The name of the custom event
    :type name: str
    """
    _attribute_map = {'name': {'key': 'name', 'type': 'str'}}

    def __init__(self, *, name: str=None, **kwargs) -> None:
        if False:
            print('Hello World!')
        super(EventsCustomEventInfo, self).__init__(**kwargs)
        self.name = name