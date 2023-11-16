from seleniumbase import BaseCase
BaseCase.main(__name__, __file__)

class PdfAssertTests(BaseCase):

    def test_assert_pdf_text(self):
        if False:
            i = 10
            return i + 15
        self.open('data:,')
        self.assert_pdf_text('https://nostarch.com/download/Automate_the_Boring_Stuff_dTOC.pdf', 'Programming Is a Creative Activity', page=1)
        self.assert_pdf_text('https://nostarch.com/download/Automate_the_Boring_Stuff_dTOC.pdf', 'Extracting Text from PDFs')