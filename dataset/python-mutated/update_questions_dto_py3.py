from msrest.serialization import Model

class UpdateQuestionsDTO(Model):
    """PATCH Body schema for Update Kb which contains list of questions to be
    added and deleted.

    :param add: List of questions to be added
    :type add: list[str]
    :param delete: List of questions to be deleted.
    :type delete: list[str]
    """
    _attribute_map = {'add': {'key': 'add', 'type': '[str]'}, 'delete': {'key': 'delete', 'type': '[str]'}}

    def __init__(self, *, add=None, delete=None, **kwargs) -> None:
        if False:
            print('Hello World!')
        super(UpdateQuestionsDTO, self).__init__(**kwargs)
        self.add = add
        self.delete = delete