from msrest.serialization import Model

class AnswerSpanResponseDTO(Model):
    """Answer span object of QnA.

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
            for i in range(10):
                print('nop')
        super(AnswerSpanResponseDTO, self).__init__(**kwargs)
        self.text = text
        self.score = score
        self.start_index = start_index
        self.end_index = end_index