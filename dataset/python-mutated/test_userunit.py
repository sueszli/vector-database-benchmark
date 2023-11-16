from __future__ import annotations
from math import isclose
import pytest
from ocrmypdf.exceptions import ExitCode
from ocrmypdf.pdfinfo import PdfInfo
from .conftest import check_ocrmypdf, run_ocrmypdf_api

@pytest.fixture
def poster(resources):
    if False:
        i = 10
        return i + 15
    return resources / 'poster.pdf'

@pytest.mark.parametrize('mode', ['pdf', 'pdfa'])
def test_userunit_pdf_passes(mode, poster, outpdf):
    if False:
        return 10
    before = PdfInfo(poster)
    check_ocrmypdf(poster, outpdf, f'--output-type={mode}', '--plugin', 'tests/plugins/tesseract_cache.py')
    after = PdfInfo(outpdf)
    assert isclose(before[0].width_inches, after[0].width_inches)

def test_rotate_interaction(poster, outpdf):
    if False:
        return 10
    check_ocrmypdf(poster, outpdf, '--output-type=pdf', '--rotate-pages', '--plugin', 'tests/plugins/tesseract_cache.py')