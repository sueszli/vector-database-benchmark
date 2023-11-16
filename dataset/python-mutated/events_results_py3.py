from msrest.serialization import Model

class EventsResults(Model):
    """An events query result.

    :param odatacontext: OData context metadata endpoint for this response
    :type odatacontext: str
    :param aimessages: OData messages for this response.
    :type aimessages: list[~azure.applicationinsights.models.ErrorInfo]
    :param value: Contents of the events query result.
    :type value: list[~azure.applicationinsights.models.EventsResultData]
    """
    _attribute_map = {'odatacontext': {'key': '@odata\\.context', 'type': 'str'}, 'aimessages': {'key': '@ai\\.messages', 'type': '[ErrorInfo]'}, 'value': {'key': 'value', 'type': '[EventsResultData]'}}

    def __init__(self, *, odatacontext: str=None, aimessages=None, value=None, **kwargs) -> None:
        if False:
            return 10
        super(EventsResults, self).__init__(**kwargs)
        self.odatacontext = odatacontext
        self.aimessages = aimessages
        self.value = value