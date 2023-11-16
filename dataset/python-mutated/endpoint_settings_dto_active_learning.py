from .active_learning_settings_dto import ActiveLearningSettingsDTO

class EndpointSettingsDTOActiveLearning(ActiveLearningSettingsDTO):
    """Active Learning settings of the endpoint.

    :param enable: True/False string providing Active Learning
    :type enable: str
    """
    _attribute_map = {'enable': {'key': 'enable', 'type': 'str'}}

    def __init__(self, **kwargs):
        if False:
            return 10
        super(EndpointSettingsDTOActiveLearning, self).__init__(**kwargs)