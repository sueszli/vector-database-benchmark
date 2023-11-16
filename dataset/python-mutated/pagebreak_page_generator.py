__license__ = 'GPL v3'
__copyright__ = '2022, Vaso Peras-Likodric <vaso at vipl.in.rs>'
__docformat__ = 'restructuredtext en'
from typing import Optional
from calibre.devices.kindle.apnx_page_generator.generators.fast_page_generator import FastPageGenerator
from calibre.devices.kindle.apnx_page_generator.i_page_generator import IPageGenerator, mobi_html
from calibre.devices.kindle.apnx_page_generator.pages import Pages
import re

class PagebreakPageGenerator(IPageGenerator):

    def name(self) -> str:
        if False:
            print('Hello World!')
        return 'pagebreak'

    def _generate_fallback(self, mobi_file_path: str, real_count: Optional[int]) -> Pages:
        if False:
            while True:
                i = 10
        return FastPageGenerator.instance.generate(mobi_file_path, real_count)

    def _generate(self, mobi_file_path: str, real_count: Optional[int]) -> Pages:
        if False:
            print('Hello World!')
        ' Determine pages based on the presence of <*pagebreak*/>. '
        html = mobi_html(mobi_file_path)
        pages = []
        for m in re.finditer(b'<[^>]*pagebreak[^>]*>', html):
            pages.append(m.end())
        return Pages(pages)
PagebreakPageGenerator.instance = PagebreakPageGenerator()