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

    def __init__(self, *, id: str=None, account_id: str=None, authenticated_id: str=None, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(EventsUserInfo, self).__init__(**kwargs)
        self.id = id
        self.account_id = account_id
        self.authenticated_id = authenticated_id