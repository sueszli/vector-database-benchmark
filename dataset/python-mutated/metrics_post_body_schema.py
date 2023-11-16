from msrest.serialization import Model

class MetricsPostBodySchema(Model):
    """A metric request.

    All required parameters must be populated in order to send to Azure.

    :param id: Required. An identifier for this query.  Must be unique within
     the post body of the request.  This identifier will be the 'id' property
     of the response object representing this query.
    :type id: str
    :param parameters: Required. The parameters for a single metrics query
    :type parameters:
     ~azure.applicationinsights.models.MetricsPostBodySchemaParameters
    """
    _validation = {'id': {'required': True}, 'parameters': {'required': True}}
    _attribute_map = {'id': {'key': 'id', 'type': 'str'}, 'parameters': {'key': 'parameters', 'type': 'MetricsPostBodySchemaParameters'}}

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(MetricsPostBodySchema, self).__init__(**kwargs)
        self.id = kwargs.get('id', None)
        self.parameters = kwargs.get('parameters', None)