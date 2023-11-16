"""
A file attachment annotation (PDF 1.3) contains a reference to a file, which typically shall be embedded in the
PDF file (see 7.11.4, “Embedded File Streams”).
"""
from borb.pdf.canvas.layout.annotation.annotation import Annotation

class FileAttachmentAnnotation(Annotation):
    """
    A file attachment annotation (PDF 1.3) contains a reference to a file, which typically shall be embedded in the
    PDF file (see 7.11.4, “Embedded File Streams”).
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(FileAttachmentAnnotation, self).__init__()