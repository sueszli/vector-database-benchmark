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

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super(DocumentStatistics, self).__init__(**kwargs)
        self.characters_count = kwargs.get('characters_count', None)
        self.transactions_count = kwargs.get('transactions_count', None)