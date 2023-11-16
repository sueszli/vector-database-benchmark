from msrest.serialization import Model

class EventsResultDataCustomDimensions(Model):
    """Custom dimensions of the event.

    :param additional_properties:
    :type additional_properties: object
    """
    _attribute_map = {'additional_properties': {'key': 'additionalProperties', 'type': 'object'}}

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(EventsResultDataCustomDimensions, self).__init__(**kwargs)
        self.additional_properties = kwargs.get('additional_properties', None)