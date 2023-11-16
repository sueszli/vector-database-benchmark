from msrest.serialization import Model

class EventsResultDataCustomMeasurements(Model):
    """Custom measurements of the event.

    :param additional_properties:
    :type additional_properties: object
    """
    _attribute_map = {'additional_properties': {'key': 'additionalProperties', 'type': 'object'}}

    def __init__(self, *, additional_properties=None, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super(EventsResultDataCustomMeasurements, self).__init__(**kwargs)
        self.additional_properties = additional_properties