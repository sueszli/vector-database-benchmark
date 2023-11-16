from msrest.serialization import Model

class ImageTagRegion(Model):
    """Defines an image region relevant to the ImageTag.

    All required parameters must be populated in order to send to Azure.

    :param query_rectangle: Required. A rectangle that outlines the area of
     interest for this tag.
    :type query_rectangle:
     ~azure.cognitiveservices.search.visualsearch.models.NormalizedQuadrilateral
    :param display_rectangle: Required. A recommended rectangle to show to the
     user.
    :type display_rectangle:
     ~azure.cognitiveservices.search.visualsearch.models.NormalizedQuadrilateral
    """
    _validation = {'query_rectangle': {'required': True}, 'display_rectangle': {'required': True}}
    _attribute_map = {'query_rectangle': {'key': 'queryRectangle', 'type': 'NormalizedQuadrilateral'}, 'display_rectangle': {'key': 'displayRectangle', 'type': 'NormalizedQuadrilateral'}}

    def __init__(self, *, query_rectangle, display_rectangle, **kwargs) -> None:
        if False:
            return 10
        super(ImageTagRegion, self).__init__(**kwargs)
        self.query_rectangle = query_rectangle
        self.display_rectangle = display_rectangle