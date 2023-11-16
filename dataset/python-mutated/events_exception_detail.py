from msrest.serialization import Model

class EventsExceptionDetail(Model):
    """Exception details.

    :param severity_level: The severity level of the exception detail
    :type severity_level: str
    :param outer_id: The outer ID of the exception detail
    :type outer_id: str
    :param message: The message of the exception detail
    :type message: str
    :param type: The type of the exception detail
    :type type: str
    :param id: The ID of the exception detail
    :type id: str
    :param parsed_stack: The parsed stack
    :type parsed_stack:
     list[~azure.applicationinsights.models.EventsExceptionDetailsParsedStack]
    """
    _attribute_map = {'severity_level': {'key': 'severityLevel', 'type': 'str'}, 'outer_id': {'key': 'outerId', 'type': 'str'}, 'message': {'key': 'message', 'type': 'str'}, 'type': {'key': 'type', 'type': 'str'}, 'id': {'key': 'id', 'type': 'str'}, 'parsed_stack': {'key': 'parsedStack', 'type': '[EventsExceptionDetailsParsedStack]'}}

    def __init__(self, **kwargs):
        if False:
            return 10
        super(EventsExceptionDetail, self).__init__(**kwargs)
        self.severity_level = kwargs.get('severity_level', None)
        self.outer_id = kwargs.get('outer_id', None)
        self.message = kwargs.get('message', None)
        self.type = kwargs.get('type', None)
        self.id = kwargs.get('id', None)
        self.parsed_stack = kwargs.get('parsed_stack', None)