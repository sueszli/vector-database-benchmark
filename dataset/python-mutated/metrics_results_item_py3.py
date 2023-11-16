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

    def __init__(self, *, id: str, status: int, body, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super(MetricsResultsItem, self).__init__(**kwargs)
        self.id = id
        self.status = status
        self.body = body