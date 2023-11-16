"""
A trap network annotation (PDF 1.3) may be used to define the trapping characteristics for a page of a PDF
document.
"""
from borb.pdf.canvas.layout.annotation.annotation import Annotation

class TrapNetAnnotation(Annotation):
    """
    A trap network annotation (PDF 1.3) may be used to define the trapping characteristics for a page of a PDF
    document.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(TrapNetAnnotation, self).__init__()