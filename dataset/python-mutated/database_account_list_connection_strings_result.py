from msrest.serialization import Model

class DatabaseAccountListConnectionStringsResult(Model):
    """The connection strings for the given database account.

    :param connection_strings: An array that contains the connection strings
     for the DocumentDB account.
    :type connection_strings: list of :class:`DatabaseAccountConnectionString
     <azure.mgmt.documentdb.models.DatabaseAccountConnectionString>`
    """
    _attribute_map = {'connection_strings': {'key': 'connectionStrings', 'type': '[DatabaseAccountConnectionString]'}}

    def __init__(self, connection_strings=None):
        if False:
            i = 10
            return i + 15
        self.connection_strings = connection_strings