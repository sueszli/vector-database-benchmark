"""Functions that can be used for the most common use-cases for pdfminer.six"""
import logging
import sys
from io import StringIO
from typing import Any, BinaryIO, Container, Iterator, Optional, cast
from .converter import XMLConverter, HTMLConverter, TextConverter, PDFPageAggregator, HOCRConverter
from .image import ImageWriter
from .layout import LAParams, LTPage
from .pdfdevice import PDFDevice, TagExtractor
from .pdfinterp import PDFResourceManager, PDFPageInterpreter
from .pdfpage import PDFPage
from .utils import open_filename, FileOrName, AnyIO

def extract_text_to_fp(inf: BinaryIO, outfp: AnyIO, output_type: str='text', codec: str='utf-8', laparams: Optional[LAParams]=None, maxpages: int=0, page_numbers: Optional[Container[int]]=None, password: str='', scale: float=1.0, rotation: int=0, layoutmode: str='normal', output_dir: Optional[str]=None, strip_control: bool=False, debug: bool=False, disable_caching: bool=False, **kwargs: Any) -> None:
    if False:
        print('Hello World!')
    "Parses text from inf-file and writes to outfp file-like object.\n\n    Takes loads of optional arguments but the defaults are somewhat sane.\n    Beware laparams: Including an empty LAParams is not the same as passing\n    None!\n\n    :param inf: a file-like object to read PDF structure from, such as a\n        file handler (using the builtin `open()` function) or a `BytesIO`.\n    :param outfp: a file-like object to write the text to.\n    :param output_type: May be 'text', 'xml', 'html', 'hocr', 'tag'.\n        Only 'text' works properly.\n    :param codec: Text decoding codec\n    :param laparams: An LAParams object from pdfminer.layout. Default is None\n        but may not layout correctly.\n    :param maxpages: How many pages to stop parsing after\n    :param page_numbers: zero-indexed page numbers to operate on.\n    :param password: For encrypted PDFs, the password to decrypt.\n    :param scale: Scale factor\n    :param rotation: Rotation factor\n    :param layoutmode: Default is 'normal', see\n        pdfminer.converter.HTMLConverter\n    :param output_dir: If given, creates an ImageWriter for extracted images.\n    :param strip_control: Does what it says on the tin\n    :param debug: Output more logging data\n    :param disable_caching: Does what it says on the tin\n    :param other:\n    :return: nothing, acting as it does on two streams. Use StringIO to get\n        strings.\n    "
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    imagewriter = None
    if output_dir:
        imagewriter = ImageWriter(output_dir)
    rsrcmgr = PDFResourceManager(caching=not disable_caching)
    device: Optional[PDFDevice] = None
    if output_type != 'text' and outfp == sys.stdout:
        outfp = sys.stdout.buffer
    if output_type == 'text':
        device = TextConverter(rsrcmgr, outfp, codec=codec, laparams=laparams, imagewriter=imagewriter)
    elif output_type == 'xml':
        device = XMLConverter(rsrcmgr, outfp, codec=codec, laparams=laparams, imagewriter=imagewriter, stripcontrol=strip_control)
    elif output_type == 'html':
        device = HTMLConverter(rsrcmgr, outfp, codec=codec, scale=scale, layoutmode=layoutmode, laparams=laparams, imagewriter=imagewriter)
    elif output_type == 'hocr':
        device = HOCRConverter(rsrcmgr, outfp, codec=codec, laparams=laparams, stripcontrol=strip_control)
    elif output_type == 'tag':
        device = TagExtractor(rsrcmgr, cast(BinaryIO, outfp), codec=codec)
    else:
        msg = f'Output type can be text, html, xml or tag but is {output_type}'
        raise ValueError(msg)
    assert device is not None
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.get_pages(inf, page_numbers, maxpages=maxpages, password=password, caching=not disable_caching):
        page.rotate = (page.rotate + rotation) % 360
        interpreter.process_page(page)
    device.close()

def extract_text(pdf_file: FileOrName, password: str='', page_numbers: Optional[Container[int]]=None, maxpages: int=0, caching: bool=True, codec: str='utf-8', laparams: Optional[LAParams]=None) -> str:
    if False:
        while True:
            i = 10
    'Parse and return the text contained in a PDF file.\n\n    :param pdf_file: Either a file path or a file-like object for the PDF file\n        to be worked on.\n    :param password: For encrypted PDFs, the password to decrypt.\n    :param page_numbers: List of zero-indexed page numbers to extract.\n    :param maxpages: The maximum number of pages to parse\n    :param caching: If resources should be cached\n    :param codec: Text decoding codec\n    :param laparams: An LAParams object from pdfminer.layout. If None, uses\n        some default settings that often work well.\n    :return: a string containing all of the text extracted.\n    '
    if laparams is None:
        laparams = LAParams()
    with open_filename(pdf_file, 'rb') as fp, StringIO() as output_string:
        fp = cast(BinaryIO, fp)
        rsrcmgr = PDFResourceManager(caching=caching)
        device = TextConverter(rsrcmgr, output_string, codec=codec, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.get_pages(fp, page_numbers, maxpages=maxpages, password=password, caching=caching):
            interpreter.process_page(page)
        return output_string.getvalue()

def extract_pages(pdf_file: FileOrName, password: str='', page_numbers: Optional[Container[int]]=None, maxpages: int=0, caching: bool=True, laparams: Optional[LAParams]=None) -> Iterator[LTPage]:
    if False:
        while True:
            i = 10
    'Extract and yield LTPage objects\n\n    :param pdf_file: Either a file path or a file-like object for the PDF file\n        to be worked on.\n    :param password: For encrypted PDFs, the password to decrypt.\n    :param page_numbers: List of zero-indexed page numbers to extract.\n    :param maxpages: The maximum number of pages to parse\n    :param caching: If resources should be cached\n    :param laparams: An LAParams object from pdfminer.layout. If None, uses\n        some default settings that often work well.\n    :return: LTPage objects\n    '
    if laparams is None:
        laparams = LAParams()
    with open_filename(pdf_file, 'rb') as fp:
        fp = cast(BinaryIO, fp)
        resource_manager = PDFResourceManager(caching=caching)
        device = PDFPageAggregator(resource_manager, laparams=laparams)
        interpreter = PDFPageInterpreter(resource_manager, device)
        for page in PDFPage.get_pages(fp, page_numbers, maxpages=maxpages, password=password, caching=caching):
            interpreter.process_page(page)
            layout = device.get_result()
            yield layout