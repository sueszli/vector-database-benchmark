from .update_questions_dto import UpdateQuestionsDTO

class UpdateQnaDTOQuestions(UpdateQuestionsDTO):
    """List of questions associated with the answer.

    :param add: List of questions to be added
    :type add: list[str]
    :param delete: List of questions to be deleted.
    :type delete: list[str]
    """
    _attribute_map = {'add': {'key': 'add', 'type': '[str]'}, 'delete': {'key': 'delete', 'type': '[str]'}}

    def __init__(self, **kwargs):
        if False:
            return 10
        super(UpdateQnaDTOQuestions, self).__init__(**kwargs)