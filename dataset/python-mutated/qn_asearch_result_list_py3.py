from msrest.serialization import Model

class QnASearchResultList(Model):
    """Represents List of Question Answers.

    :param answers: Represents Search Result list.
    :type answers:
     list[~azure.cognitiveservices.knowledge.qnamaker.models.QnASearchResult]
    """
    _attribute_map = {'answers': {'key': 'answers', 'type': '[QnASearchResult]'}}

    def __init__(self, *, answers=None, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super(QnASearchResultList, self).__init__(**kwargs)
        self.answers = answers