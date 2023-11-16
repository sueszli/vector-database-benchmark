from msrest.serialization import Model

class FeedbackRecordDTO(Model):
    """Active learning feedback record.

    :param user_id: Unique identifier for the user.
    :type user_id: str
    :param user_question: The suggested question being provided as feedback.
    :type user_question: str
    :param qna_id: The qnaId for which the suggested question is provided as
     feedback.
    :type qna_id: int
    """
    _validation = {'user_question': {'max_length': 1000}}
    _attribute_map = {'user_id': {'key': 'userId', 'type': 'str'}, 'user_question': {'key': 'userQuestion', 'type': 'str'}, 'qna_id': {'key': 'qnaId', 'type': 'int'}}

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(FeedbackRecordDTO, self).__init__(**kwargs)
        self.user_id = kwargs.get('user_id', None)
        self.user_question = kwargs.get('user_question', None)
        self.qna_id = kwargs.get('qna_id', None)