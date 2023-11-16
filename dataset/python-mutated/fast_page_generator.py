__license__ = 'GPL v3'
__copyright__ = '2022, Vaso Peras-Likodric <vaso at vipl.in.rs>'
__docformat__ = 'restructuredtext en'
from typing import Optional
from calibre.devices.kindle.apnx_page_generator.i_page_generator import IPageGenerator, mobi_html_length
from calibre.devices.kindle.apnx_page_generator.pages import Pages

class FastPageGenerator(IPageGenerator):

    def name(self) -> str:
        if False:
            while True:
                i = 10
        return 'fast'

    def _generate_fallback(self, mobi_file_path: str, real_count: Optional[int]) -> Pages:
        if False:
            i = 10
            return i + 15
        raise Exception('Fast calculation impossible.')

    def _generate(self, mobi_file_path: str, real_count: Optional[int]) -> Pages:
        if False:
            print('Hello World!')
        "\n        2300 characters of uncompressed text per page. This is\n        not meant to map 1 to 1 to a print book but to be a\n        close enough measure.\n\n        A test book was chosen and the characters were counted\n        on one page. This number was round to 2240 then 60\n        characters of markup were added to the total giving\n        2300.\n\n        Uncompressed text length is used because it's easily\n        accessible in MOBI files (part of the header). Also,\n        It's faster to work off of the length then to\n        decompress and parse the actual text.\n        "
        pages = []
        count = 0
        text_length = mobi_html_length(mobi_file_path)
        while count < text_length:
            pages.append(count)
            count += 2300
        return Pages(pages)
FastPageGenerator.instance = FastPageGenerator()