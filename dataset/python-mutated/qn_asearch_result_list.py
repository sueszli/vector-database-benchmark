from msrest.serialization import Model

class QnASearchResultList(Model):
    """Represents List of Question Answers.

    :param answers: Represents Search Result list.
    :type answers:
     list[~azure.cognitiveservices.knowledge.qnamaker.models.QnASearchResult]
    """
    _attribute_map = {'answers': {'key': 'answers', 'type': '[QnASearchResult]'}}

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(QnASearchResultList, self).__init__(**kwargs)
        self.answers = kwargs.get('answers', None)