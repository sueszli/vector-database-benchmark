from msrest.serialization import Model

class RequestStatistics(Model):
    """RequestStatistics.

    :param documents_count: Number of documents submitted in the request.
    :type documents_count: int
    :param valid_documents_count: Number of valid documents. This excludes
     empty, over-size limit or non-supported languages documents.
    :type valid_documents_count: int
    :param erroneous_documents_count: Number of invalid documents. This
     includes empty, over-size limit or non-supported languages documents.
    :type erroneous_documents_count: int
    :param transactions_count: Number of transactions for the request.
    :type transactions_count: long
    """
    _attribute_map = {'documents_count': {'key': 'documentsCount', 'type': 'int'}, 'valid_documents_count': {'key': 'validDocumentsCount', 'type': 'int'}, 'erroneous_documents_count': {'key': 'erroneousDocumentsCount', 'type': 'int'}, 'transactions_count': {'key': 'transactionsCount', 'type': 'long'}}

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(RequestStatistics, self).__init__(**kwargs)
        self.documents_count = kwargs.get('documents_count', None)
        self.valid_documents_count = kwargs.get('valid_documents_count', None)
        self.erroneous_documents_count = kwargs.get('erroneous_documents_count', None)
        self.transactions_count = kwargs.get('transactions_count', None)