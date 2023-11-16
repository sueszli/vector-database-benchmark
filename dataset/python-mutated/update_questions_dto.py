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

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(UpdateQuestionsDTO, self).__init__(**kwargs)
        self.add = kwargs.get('add', None)
        self.delete = kwargs.get('delete', None)