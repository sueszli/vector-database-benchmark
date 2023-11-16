from msrest.serialization import Model

class QueryContextDTO(Model):
    """Context object with previous QnA's information.

    :param previous_qna_id: Previous QnA Id - qnaId of the top result.
    :type previous_qna_id: int
    :param previous_user_query: Previous user query.
    :type previous_user_query: str
    """
    _attribute_map = {'previous_qna_id': {'key': 'previousQnaId', 'type': 'int'}, 'previous_user_query': {'key': 'previousUserQuery', 'type': 'str'}}

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(QueryContextDTO, self).__init__(**kwargs)
        self.previous_qna_id = kwargs.get('previous_qna_id', None)
        self.previous_user_query = kwargs.get('previous_user_query', None)