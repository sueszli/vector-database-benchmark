from msrest.serialization import Model

class EventsTraceInfo(Model):
    """The trace information.

    :param message: The trace message
    :type message: str
    :param severity_level: The trace severity level
    :type severity_level: int
    """
    _attribute_map = {'message': {'key': 'message', 'type': 'str'}, 'severity_level': {'key': 'severityLevel', 'type': 'int'}}

    def __init__(self, *, message: str=None, severity_level: int=None, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super(EventsTraceInfo, self).__init__(**kwargs)
        self.message = message
        self.severity_level = severity_level