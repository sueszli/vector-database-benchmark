from msrest.serialization import Model

class MetricsResult(Model):
    """A metric result.

    :param value:
    :type value: ~azure.applicationinsights.models.MetricsResultInfo
    """
    _attribute_map = {'value': {'key': 'value', 'type': 'MetricsResultInfo'}}

    def __init__(self, *, value=None, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(MetricsResult, self).__init__(**kwargs)
        self.value = value