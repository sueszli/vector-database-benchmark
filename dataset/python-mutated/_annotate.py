"""
Class definition and utilities for the annotation utility of the image classification toolkit
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from ...visualization import _get_client_app_path
import turicreate.toolkits._internal_utils as _tkutl
from turicreate._cython.cy_server import QuietProgress as _QuietProgress
import turicreate as __tc

def annotate(data, image_column=None, annotation_column='annotations'):
    if False:
        while True:
            i = 10
    '\n    Annotate images using a GUI assisted application. When the GUI is\n    terminated an SFrame with the representative images and annotations is\n    returned.\n\n    Parameters\n    ----------\n    data : SArray | SFrame\n        The data containing the input images.\n\n    image_column: string, optional\n        The name of the input column in the SFrame that contains the image that\n        needs to be annotated. In case `data` is of type SArray, then the\n        output SFrame contains a column (with this name) containing the input\n        images.\n\n    annotation_column : string, optional\n        The column containing the annotations in the output SFrame.\n\n    Returns\n    -------\n    out : SFrame\n        A new SFrame that contains the newly annotated data.\n\n    Examples\n    --------\n    >>> import turicreate as tc\n    >>> images = tc.image_analysis.load_images("path/to/images")\n    >>> print(images)\n        +------------------------+--------------------------+\n        |          path          |          image           |\n        +------------------------+--------------------------+\n        | /Users/username/Doc... | Height: 1712 Width: 1952 |\n        | /Users/username/Doc... | Height: 1386 Width: 1000 |\n        | /Users/username/Doc... |  Height: 536 Width: 858  |\n        | /Users/username/Doc... | Height: 1512 Width: 2680 |\n        +------------------------+--------------------------+\n        [4 rows x 2 columns]\n\n    >>> images = tc.image_classifier.annotate(images)\n    >>> print(images)\n        +------------------------+--------------------------+-------------------+\n        |          path          |          image           |    annotations    |\n        +------------------------+--------------------------+-------------------+\n        | /Users/username/Doc... | Height: 1712 Width: 1952 |        dog        |\n        | /Users/username/Doc... | Height: 1386 Width: 1000 |        dog        |\n        | /Users/username/Doc... |  Height: 536 Width: 858  |        cat        |\n        | /Users/username/Doc... | Height: 1512 Width: 2680 |       mouse       |\n        +------------------------+--------------------------+-------------------+\n        [4 rows x 3 columns]\n\n    '
    if not isinstance(data, __tc.SFrame):
        raise TypeError('"data" must be of type SFrame.')
    if data.num_rows() == 0:
        raise Exception('input data cannot be empty')
    if image_column == None:
        image_column = _tkutl._find_only_image_column(data)
    if image_column == None:
        raise ValueError("'image_column' cannot be 'None'")
    if type(image_column) != str:
        raise TypeError("'image_column' has to be of type 'str'")
    if annotation_column == None:
        annotation_column = ''
    if type(annotation_column) != str:
        raise TypeError("'annotation_column' has to be of type 'str'")
    if type(data) == __tc.data_structures.image.Image:
        data = __tc.SFrame({image_column: __tc.SArray([data])})
    elif type(data) == __tc.data_structures.sframe.SFrame:
        if data.shape[0] == 0:
            return data
        if not data[image_column].dtype == __tc.data_structures.image.Image:
            raise TypeError("'data[image_column]' must be an SFrame or SArray")
    elif type(data) == __tc.data_structures.sarray.SArray:
        if data.shape[0] == 0:
            return data
        data = __tc.SFrame({image_column: data})
    else:
        raise TypeError("'data' must be an SFrame or SArray")
    annotation_window = __tc.extensions.create_image_classification_annotation(data, [image_column], annotation_column)
    with _QuietProgress(False):
        annotation_window.annotate(_get_client_app_path())
        return annotation_window.returnAnnotations()

def recover_annotation():
    if False:
        for i in range(10):
            print('nop')
    '\n    Recover the last annotated SFrame.\n\n    If you annotate an SFrame and forget to assign it to a variable, this\n    function allows you to recover the last annotated SFrame.\n\n    Returns\n    -------\n    out : SFrame\n        A new SFrame that contains the recovered annotation data.\n\n    Examples\n    --------\n    >>> annotations = tc.image_classifier.recover_annotation()\n    >>> print(annotations)\n    +----------------------+-------------+\n    |        images        | annotations |\n    +----------------------+-------------+\n    | Height: 28 Width: 28 |     Cat     |\n    | Height: 28 Width: 28 |     Dog     |\n    | Height: 28 Width: 28 |    Mouse    |\n    | Height: 28 Width: 28 |   Feather   |\n    | Height: 28 Width: 28 |     Bird    |\n    | Height: 28 Width: 28 |     Cat     |\n    | Height: 28 Width: 28 |     Cat     |\n    | Height: 28 Width: 28 |     Dog     |\n    | Height: 28 Width: 28 |     Cat     |\n    | Height: 28 Width: 28 |     Bird    |\n    +----------------------+-------------+\n    [400 rows x 3 columns]\n\n    '
    empty_instance = __tc.extensions.ImageClassification()
    annotation_wrapper = empty_instance.get_annotation_registry()
    return annotation_wrapper.annotation_sframe