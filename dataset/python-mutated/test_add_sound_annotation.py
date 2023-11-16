import unittest
from decimal import Decimal
from pathlib import Path
import requests
from borb.pdf.canvas.geometry.rectangle import Rectangle
from borb.pdf.canvas.layout.annotation.sound_annotation import SoundAnnotation
from borb.pdf.canvas.layout.page_layout.multi_column_layout import SingleColumnLayout
from borb.pdf.document.document import Document
from borb.pdf.page.page import Page
from borb.pdf.pdf import PDF
from tests.test_case import TestCase
unittest.TestLoader.sortTestMethodsUsing = None

class TestAddSoundAnnotation(TestCase):

    def test_add_sound_annotation(self):
        if False:
            print('Hello World!')
        mp3_file: Path = self.get_first_output_file().parent / 'audio.mp3'
        with open(mp3_file, 'wb') as fh:
            fh.write(requests.get('http://aux.incompetech.com/music/royalty-free/mp3-royaltyfree/Jazz%20Brunch.mp3', allow_redirects=True).content)
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds a RemoteGoToAnnotation to a PDF.'))
        w: Decimal = pdf.get_page(0).get_page_info().get_width()
        h: Decimal = pdf.get_page(0).get_page_info().get_height()
        pdf.get_page(0).add_annotation(SoundAnnotation(bounding_box=Rectangle(w / Decimal(2) - Decimal(32), h / Decimal(2) - Decimal(32), Decimal(64), Decimal(64)), url_to_mp3_file=str(mp3_file.absolute())))
        with open(self.get_first_output_file(), 'wb') as out_file_handle:
            PDF.dumps(out_file_handle, pdf)
        self.compare_visually_to_ground_truth(self.get_first_output_file())
        self.check_pdf_using_validator(self.get_first_output_file())