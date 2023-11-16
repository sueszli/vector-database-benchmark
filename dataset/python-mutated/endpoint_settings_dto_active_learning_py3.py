from .active_learning_settings_dto_py3 import ActiveLearningSettingsDTO

class EndpointSettingsDTOActiveLearning(ActiveLearningSettingsDTO):
    """Active Learning settings of the endpoint.

    :param enable: True/False string providing Active Learning
    :type enable: str
    """
    _attribute_map = {'enable': {'key': 'enable', 'type': 'str'}}

    def __init__(self, *, enable: str=None, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super(EndpointSettingsDTOActiveLearning, self).__init__(enable=enable, **kwargs)