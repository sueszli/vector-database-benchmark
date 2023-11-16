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

    def __init__(self, *, value: float=None, name: str=None, category: str=None, counter: str=None, instance_name: str=None, instance: str=None, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(EventsPerformanceCounterInfo, self).__init__(**kwargs)
        self.value = value
        self.name = name
        self.category = category
        self.counter = counter
        self.instance_name = instance_name
        self.instance = instance