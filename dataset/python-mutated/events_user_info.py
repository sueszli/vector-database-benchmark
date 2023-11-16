from msrest.serialization import Model

class EventsUserInfo(Model):
    """User info for an event result.

    :param id: ID of the user
    :type id: str
    :param account_id: Account ID of the user
    :type account_id: str
    :param authenticated_id: Authenticated ID of the user
    :type authenticated_id: str
    """
    _attribute_map = {'id': {'key': 'id', 'type': 'str'}, 'account_id': {'key': 'accountId', 'type': 'str'}, 'authenticated_id': {'key': 'authenticatedId', 'type': 'str'}}

    def __init__(self, **kwargs):
        if False:
            return 10
        super(EventsUserInfo, self).__init__(**kwargs)
        self.id = kwargs.get('id', None)
        self.account_id = kwargs.get('account_id', None)
        self.authenticated_id = kwargs.get('authenticated_id', None)