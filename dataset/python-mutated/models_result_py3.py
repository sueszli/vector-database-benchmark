from msrest.serialization import Model

class ModelsResult(Model):
    """Result of query operation to fetch multiple models.

    :param models_property: Collection of models.
    :type models_property:
     list[~azure.cognitiveservices.formrecognizer.models.ModelResult]
    """
    _attribute_map = {'models_property': {'key': 'models', 'type': '[ModelResult]'}}

    def __init__(self, *, models_property=None, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super(ModelsResult, self).__init__(**kwargs)
        self.models_property = models_property