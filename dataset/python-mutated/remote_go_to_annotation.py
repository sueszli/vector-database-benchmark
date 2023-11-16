"""
A link annotation represents either a hypertext link to a destination elsewhere in the document (see 12.3.2,
“Destinations”) or an action to be performed (12.6, “Actions”). Table 173 shows the annotation dictionary
entries specific to this type of annotation.
This method adds a link annotation with an action that opens a remote URI.
"""
from borb.io.read.types import Decimal as bDecimal
from borb.io.read.types import Dictionary
from borb.io.read.types import List
from borb.io.read.types import Name
from borb.io.read.types import String
from borb.pdf.canvas.geometry.rectangle import Rectangle
from borb.pdf.canvas.layout.annotation.annotation import Annotation

class RemoteGoToAnnotation(Annotation):
    """
    A link annotation represents either a hypertext link to a destination elsewhere in the document (see 12.3.2,
    “Destinations”) or an action to be performed (12.6, “Actions”). Table 173 shows the annotation dictionary
    entries specific to this type of annotation.
    This method adds a link annotation with an action that opens a remote URI.
    """

    def __init__(self, bounding_box: Rectangle, uri: str):
        if False:
            print('Hello World!')
        super(RemoteGoToAnnotation, self).__init__(bounding_box)
        self[Name('Subtype')] = Name('Link')
        self[Name('Border')] = List().set_is_inline(True)
        for _ in range(0, 3):
            self[Name('Border')].append(bDecimal(0))
        self[Name('A')] = Dictionary()
        self['A'][Name('Type')] = Name('Action')
        self['A'][Name('S')] = Name('URI')
        self['A'][Name('URI')] = String(uri)