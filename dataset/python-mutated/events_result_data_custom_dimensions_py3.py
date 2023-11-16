from msrest.serialization import Model

class EventsResultDataCustomDimensions(Model):
    """Custom dimensions of the event.

    :param additional_properties:
    :type additional_properties: object
    """
    _attribute_map = {'additional_properties': {'key': 'additionalProperties', 'type': 'object'}}

    def __init__(self, *, additional_properties=None, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super(EventsResultDataCustomDimensions, self).__init__(**kwargs)
        self.additional_properties = additional_properties