from msrest.serialization import Model

class EventsResult(Model):
    """An event query result.

    :param aimessages: OData messages for this response.
    :type aimessages: list[~azure.applicationinsights.models.ErrorInfo]
    :param value:
    :type value: ~azure.applicationinsights.models.EventsResultData
    """
    _attribute_map = {'aimessages': {'key': '@ai\\.messages', 'type': '[ErrorInfo]'}, 'value': {'key': 'value', 'type': 'EventsResultData'}}

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super(EventsResult, self).__init__(**kwargs)
        self.aimessages = kwargs.get('aimessages', None)
        self.value = kwargs.get('value', None)