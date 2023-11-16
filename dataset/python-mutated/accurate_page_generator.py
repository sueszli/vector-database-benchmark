__license__ = 'GPL v3'
__copyright__ = '2022, Vaso Peras-Likodric <vaso at vipl.in.rs>'
__docformat__ = 'restructuredtext en'
from typing import Optional
from calibre.devices.kindle.apnx_page_generator.generators.fast_page_generator import FastPageGenerator
from calibre.devices.kindle.apnx_page_generator.i_page_generator import IPageGenerator, mobi_html
from calibre.devices.kindle.apnx_page_generator.pages import Pages

class AccuratePageGenerator(IPageGenerator):
    instance = None

    def name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'accurate'

    def _generate_fallback(self, mobi_file_path: str, real_count: Optional[int]) -> Pages:
        if False:
            print('Hello World!')
        return FastPageGenerator.instance.generate(mobi_file_path, real_count)

    def _generate(self, mobi_file_path: str, real_count: Optional[int]) -> Pages:
        if False:
            print('Hello World!')
        '\n        A more accurate but much more resource intensive and slower\n        method to calculate the page length.\n\n        Parses the uncompressed text. In an average paper back book\n        There are 32 lines per page and a maximum of 70 characters\n        per line.\n\n        Each paragraph starts a new line and every 70 characters\n        (minus markup) in a paragraph starts a new line. The\n        position after every 30 lines will be marked as a new\n        page.\n\n        This can be make more accurate by accounting for\n        <div class="mbp_pagebreak" /> as a new page marker.\n        And <br> elements as an empty line.\n        '
        pages = []
        html = mobi_html(mobi_file_path)
        in_tag = False
        in_p = False
        check_p = False
        closing = False
        p_char_count = 0
        lines = []
        pos = -1
        data = bytearray(html)
        (slash, p, lt, gt) = map(ord, '/p<>')
        for c in data:
            pos += 1
            if check_p:
                if c == slash:
                    closing = True
                    continue
                elif c == p:
                    if closing:
                        in_p = False
                    else:
                        in_p = True
                        lines.append(pos - 2)
                check_p = False
                closing = False
                continue
            if c == lt:
                in_tag = True
                check_p = True
                continue
            elif c == gt:
                in_tag = False
                check_p = False
                continue
            if in_p and (not in_tag):
                p_char_count += 1
                if p_char_count == 70:
                    lines.append(pos)
                    p_char_count = 0
        for i in range(0, len(lines), 32):
            pages.append(lines[i])
        return Pages(pages)
AccuratePageGenerator.instance = AccuratePageGenerator()