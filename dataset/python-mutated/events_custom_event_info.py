from msrest.serialization import Model

class EventsCustomEventInfo(Model):
    """The custom event information.

    :param name: The name of the custom event
    :type name: str
    """
    _attribute_map = {'name': {'key': 'name', 'type': 'str'}}

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super(EventsCustomEventInfo, self).__init__(**kwargs)
        self.name = kwargs.get('name', None)