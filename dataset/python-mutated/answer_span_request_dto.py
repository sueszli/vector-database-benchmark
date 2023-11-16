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

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super(AnswerSpanRequestDTO, self).__init__(**kwargs)
        self.enable = kwargs.get('enable', None)
        self.score_threshold = kwargs.get('score_threshold', None)
        self.top_answers_with_span = kwargs.get('top_answers_with_span', None)