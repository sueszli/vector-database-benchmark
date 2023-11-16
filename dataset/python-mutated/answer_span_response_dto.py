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

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super(AnswerSpanResponseDTO, self).__init__(**kwargs)
        self.text = kwargs.get('text', None)
        self.score = kwargs.get('score', None)
        self.start_index = kwargs.get('start_index', None)
        self.end_index = kwargs.get('end_index', None)