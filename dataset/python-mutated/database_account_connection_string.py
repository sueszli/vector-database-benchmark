from msrest.serialization import Model

class DatabaseAccountConnectionString(Model):
    """Connection string for the DocumentDB account.

    Variables are only populated by the server, and will be ignored when
    sending a request.

    :ivar connection_string: Value of the connection string
    :vartype connection_string: str
    :ivar description: Description of the connection string
    :vartype description: str
    """
    _validation = {'connection_string': {'readonly': True}, 'description': {'readonly': True}}
    _attribute_map = {'connection_string': {'key': 'connectionString', 'type': 'str'}, 'description': {'key': 'description', 'type': 'str'}}

    def __init__(self):
        if False:
            while True:
                i = 10
        self.connection_string = None
        self.description = None