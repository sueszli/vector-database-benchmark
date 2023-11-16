from msrest.serialization import Model

class MetricsResultsItem(Model):
    """MetricsResultsItem.

    All required parameters must be populated in order to send to Azure.

    :param id: Required. The specified ID for this metric.
    :type id: str
    :param status: Required. The HTTP status code of this metric query.
    :type status: int
    :param body: Required. The results of this metric query.
    :type body: ~azure.applicationinsights.models.MetricsResult
    """
    _validation = {'id': {'required': True}, 'status': {'required': True}, 'body': {'required': True}}
    _attribute_map = {'id': {'key': 'id', 'type': 'str'}, 'status': {'key': 'status', 'type': 'int'}, 'body': {'key': 'body', 'type': 'MetricsResult'}}

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(MetricsResultsItem, self).__init__(**kwargs)
        self.id = kwargs.get('id', None)
        self.status = kwargs.get('status', None)
        self.body = kwargs.get('body', None)