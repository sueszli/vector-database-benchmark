from msrest.serialization import Model

class RecipesModule(Model):
    """Defines a list of recipes.

    Variables are only populated by the server, and will be ignored when
    sending a request.

    :ivar value: A list of recipes.
    :vartype value:
     list[~azure.cognitiveservices.search.visualsearch.models.Recipe]
    """
    _validation = {'value': {'readonly': True}}
    _attribute_map = {'value': {'key': 'value', 'type': '[Recipe]'}}

    def __init__(self, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(RecipesModule, self).__init__(**kwargs)
        self.value = None