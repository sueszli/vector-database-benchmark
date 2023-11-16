from msrest.serialization import Model

class DocumentStatistics(Model):
    """DocumentStatistics.

    :param characters_count: Number of text elements recognized in the
     document.
    :type characters_count: int
    :param transactions_count: Number of transactions for the document.
    :type transactions_count: int
    """
    _attribute_map = {'characters_count': {'key': 'charactersCount', 'type': 'int'}, 'transactions_count': {'key': 'transactionsCount', 'type': 'int'}}

    def __init__(self, *, characters_count: int=None, transactions_count: int=None, **kwargs) -> None:
        if False:
            return 10
        super(DocumentStatistics, self).__init__(**kwargs)
        self.characters_count = characters_count
        self.transactions_count = transactions_count