import typing
import unittest
from borb.pdf import Document
from borb.pdf import PDF
from tests.test_case import TestCase
unittest.TestLoader.sortTestMethodsUsing = None

class TestDocumentHasSignatures(TestCase):

    def test_document_has_signatures_001(self):
        if False:
            print('Hello World!')
        doc: typing.Optional[Document] = None
        with open(self.get_artifacts_directory() / 'hello_world_signed_initials_001.pdf', 'rb') as pdf_file_handle:
            doc = PDF.loads(pdf_file_handle)
        assert doc is not None
        assert doc.get_document_info().has_signatures() == False

    def test_document_has_signatures_002(self):
        if False:
            while True:
                i = 10
        doc: typing.Optional[Document] = None
        with open(self.get_artifacts_directory() / 'hello_world_signed_initials_002.pdf', 'rb') as pdf_file_handle:
            doc = PDF.loads(pdf_file_handle)
        assert doc is not None
        assert doc.get_document_info().has_signatures()