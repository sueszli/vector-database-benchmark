import collections
import os
import sys
from typing import Any, Counter, Iterator, List
from warnings import warn
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTContainer
from pdfminer.pdfdocument import PDFDocument, PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
warn('The file pdfstats.py will be removed in 2023. It was probably introduced for testing purposes a long time ago, and no longer relevant. Feel free to create a GitHub issue if you disagree.', DeprecationWarning)
(_, SCRIPT) = os.path.split(__file__)

def msg(*args: object, **kwargs: Any) -> None:
    if False:
        for i in range(10):
            print('nop')
    print(' '.join(map(str, args)), **kwargs)

def flat_iter(obj: object) -> Iterator[object]:
    if False:
        return 10
    yield obj
    if isinstance(obj, LTContainer):
        for ob in obj:
            yield from flat_iter(ob)

def main(args: List[str]) -> int:
    if False:
        for i in range(10):
            print('nop')
    msg(SCRIPT, args)
    if len(args) != 1:
        msg('Parse a PDF file and print some pdfminer-specific stats')
        msg('Usage:', SCRIPT, '<PDF-filename>')
        return 1
    (infilename,) = args
    lt_types: Counter[str] = collections.Counter()
    with open(infilename, 'rb') as pdf_file:
        parser = PDFParser(pdf_file)
        password = ''
        document = PDFDocument(parser, password)
        if not document.is_extractable:
            raise PDFTextExtractionNotAllowed(infilename)
        pages = PDFPage.create_pages(document)
        rsrcmgr = PDFResourceManager()
        laparams = LAParams(detect_vertical=True, all_texts=True)
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for (page_count, page) in enumerate(pages, 1):
            interpreter.process_page(page)
            layout = device.get_result()
            lt_types.update((type(item).__name__ for item in flat_iter(layout)))
    msg('page_count', page_count)
    msg('lt_types:', ' '.join(('{}:{}'.format(*tc) for tc in lt_types.items())))
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))