from msrest.serialization import Model

class EventsApplicationInfo(Model):
    """Application info for an event result.

    :param version: Version of the application
    :type version: str
    """
    _attribute_map = {'version': {'key': 'version', 'type': 'str'}}

    def __init__(self, *, version: str=None, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(EventsApplicationInfo, self).__init__(**kwargs)
        self.version = version