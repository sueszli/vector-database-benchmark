from msrest.serialization import Model

class MetricsResult(Model):
    """A metric result.

    :param value:
    :type value: ~azure.applicationinsights.models.MetricsResultInfo
    """
    _attribute_map = {'value': {'key': 'value', 'type': 'MetricsResultInfo'}}

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(MetricsResult, self).__init__(**kwargs)
        self.value = kwargs.get('value', None)