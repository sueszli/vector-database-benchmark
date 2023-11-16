from msrest.serialization import Model

class EventsPerformanceCounterInfo(Model):
    """The performance counter info.

    :param value: The value of the performance counter
    :type value: float
    :param name: The name of the performance counter
    :type name: str
    :param category: The category of the performance counter
    :type category: str
    :param counter: The counter of the performance counter
    :type counter: str
    :param instance_name: The instance name of the performance counter
    :type instance_name: str
    :param instance: The instance of the performance counter
    :type instance: str
    """
    _attribute_map = {'value': {'key': 'value', 'type': 'float'}, 'name': {'key': 'name', 'type': 'str'}, 'category': {'key': 'category', 'type': 'str'}, 'counter': {'key': 'counter', 'type': 'str'}, 'instance_name': {'key': 'instanceName', 'type': 'str'}, 'instance': {'key': 'instance', 'type': 'str'}}

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super(EventsPerformanceCounterInfo, self).__init__(**kwargs)
        self.value = kwargs.get('value', None)
        self.name = kwargs.get('name', None)
        self.category = kwargs.get('category', None)
        self.counter = kwargs.get('counter', None)
        self.instance_name = kwargs.get('instance_name', None)
        self.instance = kwargs.get('instance', None)