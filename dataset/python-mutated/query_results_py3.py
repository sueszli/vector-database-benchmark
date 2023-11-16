from msrest.serialization import Model

class QueryResults(Model):
    """A query response.

    Contains the tables, columns & rows resulting from a query.

    All required parameters must be populated in order to send to Azure.

    :param tables: Required. The list of tables, columns and rows.
    :type tables: list[~azure.applicationinsights.models.Table]
    """
    _validation = {'tables': {'required': True}}
    _attribute_map = {'tables': {'key': 'tables', 'type': '[Table]'}}

    def __init__(self, *, tables, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(QueryResults, self).__init__(**kwargs)
        self.tables = tables