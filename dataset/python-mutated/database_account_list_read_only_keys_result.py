from msrest.serialization import Model

class DatabaseAccountListReadOnlyKeysResult(Model):
    """The read-only access keys for the given database account.

    Variables are only populated by the server, and will be ignored when
    sending a request.

    :ivar primary_readonly_master_key: Base 64 encoded value of the primary
     read-only key.
    :vartype primary_readonly_master_key: str
    :ivar secondary_readonly_master_key: Base 64 encoded value of the
     secondary read-only key.
    :vartype secondary_readonly_master_key: str
    """
    _validation = {'primary_readonly_master_key': {'readonly': True}, 'secondary_readonly_master_key': {'readonly': True}}
    _attribute_map = {'primary_readonly_master_key': {'key': 'primaryReadonlyMasterKey', 'type': 'str'}, 'secondary_readonly_master_key': {'key': 'secondaryReadonlyMasterKey', 'type': 'str'}}

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.primary_readonly_master_key = None
        self.secondary_readonly_master_key = None