from msrest.paging import Paged

class DatabaseAccountPaged(Paged):
    """
    A paging container for iterating over a list of DatabaseAccount object
    """
    _attribute_map = {'next_link': {'key': 'nextLink', 'type': 'str'}, 'current_page': {'key': 'value', 'type': '[DatabaseAccount]'}}

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(DatabaseAccountPaged, self).__init__(*args, **kwargs)