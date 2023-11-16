import typing
import unittest
from pathlib import Path
from borb.pdf.pdf import PDF
from borb.toolkit.text.simple_text_extraction import SimpleTextExtraction
from tests.test_case import TestCase
unittest.TestLoader.sortTestMethodsUsing = None

class TestExtractTextUsingOCR(TestCase):

    @unittest.skip
    def test_write_ocr_as_optional_content_group(self):
        if False:
            return 10
        from borb.toolkit.ocr.ocr_as_optional_content_group import OCRAsOptionalContentGroup
        input_file: Path = self.get_artifacts_directory() / 'input_001.pdf'
        tesseract_data_dir: Path = Path.home() / Path('Downloads/tessdata-main/')
        with open(input_file, 'rb') as pdf_file_handle:
            l = OCRAsOptionalContentGroup(tesseract_data_dir)
            doc = PDF.loads(pdf_file_handle, [l])
        with open(self.get_first_output_file(), 'wb') as pdf_file_handle:
            PDF.dumps(pdf_file_handle, doc)

    @unittest.skip
    def test_read_enhanced_document(self):
        if False:
            i = 10
            return i + 15
        l = SimpleTextExtraction()
        with open(self.get_first_output_file(), 'rb') as pdf_file_handle:
            PDF.loads(pdf_file_handle, [l])
        txt: str = l.get_text_for_page(0)
        ground_truth: str = '\n        H2020 Programme\n        AGA  – Annotated Model Grant Agreement\n        Version 5.2\n        26 June 2019\n        Disclaimer\n        This guide is aimed at assisting beneficiaries. It is provided for information purposes only and is not intended\n        to replace consultation of any applicable legal sources. Neither the Commission nor the Executive Agencies (or\n        any person acting on their behalf) can be held responsible for the use made of this guidance document.\n        The EU Framework Programme\n        for Research and Innovation\n        HORIZON2020        \n        '
        letter_frequency_001: typing.Dict[str, int] = {x: sum([1 for c in ground_truth if c == x]) for x in 'abcdefghijklmnopqrstuvwxyz'}
        letter_frequency_002: typing.Dict[str, int] = {x: sum([1 for c in txt if c == x]) for x in 'abcdefghijklmnopqrstuvwxyz'}
        assert all([letter_frequency_002[k] == v for (k, v) in letter_frequency_001.items()])
if __name__ == '__main__':
    unittest.main()