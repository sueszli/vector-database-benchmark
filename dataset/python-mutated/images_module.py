from msrest.serialization import Model

class ImagesModule(Model):
    """Defines a list of images.

    Variables are only populated by the server, and will be ignored when
    sending a request.

    :ivar value: A list of images.
    :vartype value:
     list[~azure.cognitiveservices.search.visualsearch.models.ImageObject]
    """
    _validation = {'value': {'readonly': True}}
    _attribute_map = {'value': {'key': 'value', 'type': '[ImageObject]'}}

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super(ImagesModule, self).__init__(**kwargs)
        self.value = None