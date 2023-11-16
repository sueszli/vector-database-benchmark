from .answer_span_response_dto_py3 import AnswerSpanResponseDTO

class QnASearchResultAnswerSpan(AnswerSpanResponseDTO):
    """Answer span object of QnA with respect to user's question.

    :param text: Predicted text of answer span.
    :type text: str
    :param score: Predicted score of answer span.
    :type score: float
    :param start_index: Start index of answer span in answer.
    :type start_index: int
    :param end_index: End index of answer span in answer.
    :type end_index: int
    """
    _attribute_map = {'text': {'key': 'text', 'type': 'str'}, 'score': {'key': 'score', 'type': 'float'}, 'start_index': {'key': 'startIndex', 'type': 'int'}, 'end_index': {'key': 'endIndex', 'type': 'int'}}

    def __init__(self, *, text: str=None, score: float=None, start_index: int=None, end_index: int=None, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super(QnASearchResultAnswerSpan, self).__init__(text=text, score=score, start_index=start_index, end_index=end_index, **kwargs)