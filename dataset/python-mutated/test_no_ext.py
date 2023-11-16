import unittest
import os
import textract

class No_Ext_TestCase(unittest.TestCase):

    def test_docx(self):
        if False:
            while True:
                i = 10
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        docx_file = os.path.join(current_dir, 'tests/no_ext/docx_paragraphs_and_tables')
        text = textract.process(docx_file, extension='docx')
        print(text)

    def test_msg(self):
        if False:
            for i in range(10):
                print('nop')
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        msg_file = os.path.join(current_dir, 'tests/no_ext/msg_standardized_text')
        text = textract.process(msg_file, extension='msg')
        print(text)

    def test_pdf(self):
        if False:
            for i in range(10):
                print('nop')
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pdf_file = os.path.join(current_dir, 'tests/no_ext/pdf_standardized_text')
        text = textract.process(pdf_file, extension='.pdf')
        print(text)