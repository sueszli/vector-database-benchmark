from msrest.serialization import Model

class EventsApplicationInfo(Model):
    """Application info for an event result.

    :param version: Version of the application
    :type version: str
    """
    _attribute_map = {'version': {'key': 'version', 'type': 'str'}}

    def __init__(self, **kwargs):
        if False:
            return 10
        super(EventsApplicationInfo, self).__init__(**kwargs)
        self.version = kwargs.get('version', None)