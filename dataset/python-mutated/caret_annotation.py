"""
A caret annotation (PDF 1.5) is a visual symbol that indicates the presence of text edits. Table 180 lists the
entries specific to caret annotations.
"""
from borb.pdf.canvas.layout.annotation.annotation import Annotation

class CaretAnnotation(Annotation):
    """
    A caret annotation (PDF 1.5) is a visual symbol that indicates the presence of text edits. Table 180 lists the
    entries specific to caret annotations.
    """

    def __init__(self):
        if False:
            return 10
        super(CaretAnnotation, self).__init__()