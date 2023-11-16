from msrest.serialization import Model

class EndpointSettingsDTO(Model):
    """Endpoint settings.

    :param active_learning: Active Learning settings of the endpoint.
    :type active_learning:
     ~azure.cognitiveservices.knowledge.qnamaker.models.EndpointSettingsDTOActiveLearning
    """
    _attribute_map = {'active_learning': {'key': 'activeLearning', 'type': 'EndpointSettingsDTOActiveLearning'}}

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(EndpointSettingsDTO, self).__init__(**kwargs)
        self.active_learning = kwargs.get('active_learning', None)