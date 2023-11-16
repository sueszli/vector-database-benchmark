from .query_context_dto import QueryContextDTO

class QueryDTOContext(QueryContextDTO):
    """Context object with previous QnA's information.

    :param previous_qna_id: Previous QnA Id - qnaId of the top result.
    :type previous_qna_id: int
    :param previous_user_query: Previous user query.
    :type previous_user_query: str
    """
    _attribute_map = {'previous_qna_id': {'key': 'previousQnaId', 'type': 'int'}, 'previous_user_query': {'key': 'previousUserQuery', 'type': 'str'}}

    def __init__(self, **kwargs):
        if False:
            return 10
        super(QueryDTOContext, self).__init__(**kwargs)