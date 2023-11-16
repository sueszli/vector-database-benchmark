from msrest.serialization import Model

class CropArea(Model):
    """A JSON object consisting of coordinates specifying the four corners of a
    cropped rectangle within the input image.

    All required parameters must be populated in order to send to Azure.

    :param top: Required. The top coordinate of the region to be cropped. The
     coordinate is a fractional value of the original image's height and is
     measured from the top edge of the image. Specify the coordinate as a value
     from 0.0 through 1.0.
    :type top: float
    :param bottom: Required. The bottom coordinate of the region to be
     cropped. The coordinate is a fractional value of the original image's
     height and is measured from the top edge of the image. Specify the
     coordinate as a value from 0.0 through 1.0.
    :type bottom: float
    :param left: Required. The left coordinate of the region to be cropped.
     The coordinate is a fractional value of the original image's width and is
     measured from the left edge of the image. Specify the coordinate as a
     value from 0.0 through 1.0.
    :type left: float
    :param right: Required. The right coordinate of the region to be cropped.
     The coordinate is a fractional value of the original image's width and is
     measured from the left edge of the image. Specify the coordinate as a
     value from 0.0 through 1.0.
    :type right: float
    """
    _validation = {'top': {'required': True}, 'bottom': {'required': True}, 'left': {'required': True}, 'right': {'required': True}}
    _attribute_map = {'top': {'key': 'top', 'type': 'float'}, 'bottom': {'key': 'bottom', 'type': 'float'}, 'left': {'key': 'left', 'type': 'float'}, 'right': {'key': 'right', 'type': 'float'}}

    def __init__(self, **kwargs):
        if False:
            return 10
        super(CropArea, self).__init__(**kwargs)
        self.top = kwargs.get('top', None)
        self.bottom = kwargs.get('bottom', None)
        self.left = kwargs.get('left', None)
        self.right = kwargs.get('right', None)