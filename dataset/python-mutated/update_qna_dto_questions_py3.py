from .update_questions_dto_py3 import UpdateQuestionsDTO

class UpdateQnaDTOQuestions(UpdateQuestionsDTO):
    """List of questions associated with the answer.

    :param add: List of questions to be added
    :type add: list[str]
    :param delete: List of questions to be deleted.
    :type delete: list[str]
    """
    _attribute_map = {'add': {'key': 'add', 'type': '[str]'}, 'delete': {'key': 'delete', 'type': '[str]'}}

    def __init__(self, *, add=None, delete=None, **kwargs) -> None:
        if False:
            return 10
        super(UpdateQnaDTOQuestions, self).__init__(add=add, delete=delete, **kwargs)