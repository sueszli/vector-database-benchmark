from msrest.serialization import Model

class EventsExceptionDetailsParsedStack(Model):
    """A parsed stack entry.

    :param assembly: The assembly of the stack entry
    :type assembly: str
    :param method: The method of the stack entry
    :type method: str
    :param level: The level of the stack entry
    :type level: long
    :param line: The line of the stack entry
    :type line: long
    """
    _attribute_map = {'assembly': {'key': 'assembly', 'type': 'str'}, 'method': {'key': 'method', 'type': 'str'}, 'level': {'key': 'level', 'type': 'long'}, 'line': {'key': 'line', 'type': 'long'}}

    def __init__(self, **kwargs):
        if False:
            return 10
        super(EventsExceptionDetailsParsedStack, self).__init__(**kwargs)
        self.assembly = kwargs.get('assembly', None)
        self.method = kwargs.get('method', None)
        self.level = kwargs.get('level', None)
        self.line = kwargs.get('line', None)