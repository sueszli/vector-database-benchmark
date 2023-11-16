from msrest.serialization import Model

class ActiveLearningSettingsDTO(Model):
    """Active Learning settings of the endpoint.

    :param enable: True/False string providing Active Learning
    :type enable: str
    """
    _attribute_map = {'enable': {'key': 'enable', 'type': 'str'}}

    def __init__(self, **kwargs):
        if False:
            return 10
        super(ActiveLearningSettingsDTO, self).__init__(**kwargs)
        self.enable = kwargs.get('enable', None)