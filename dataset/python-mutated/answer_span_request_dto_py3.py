from msrest.serialization import Model

class AnswerSpanRequestDTO(Model):
    """To configure Answer span prediction feature.

    :param enable: Enable or Disable Answer Span prediction.
    :type enable: bool
    :param score_threshold: Minimum threshold score required to include an
     answer span.
    :type score_threshold: float
    :param top_answers_with_span: Number of Top answers to be considered for
     span prediction.
    :type top_answers_with_span: int
    """
    _validation = {'top_answers_with_span': {'maximum': 10, 'minimum': 1}}
    _attribute_map = {'enable': {'key': 'enable', 'type': 'bool'}, 'score_threshold': {'key': 'scoreThreshold', 'type': 'float'}, 'top_answers_with_span': {'key': 'topAnswersWithSpan', 'type': 'int'}}

    def __init__(self, *, enable: bool=None, score_threshold: float=None, top_answers_with_span: int=None, **kwargs) -> None:
        if False:
            print('Hello World!')
        super(AnswerSpanRequestDTO, self).__init__(**kwargs)
        self.enable = enable
        self.score_threshold = score_threshold
        self.top_answers_with_span = top_answers_with_span