from __future__ import annotations
from unittest.mock import patch
import img2pdf
import pikepdf
import pytest
from PIL import Image
import ocrmypdf
from .conftest import check_ocrmypdf, run_ocrmypdf_api

@pytest.fixture
def baiona(resources):
    if False:
        while True:
            i = 10
    return Image.open(resources / 'baiona_gray.png')

def test_image_to_pdf(resources, outpdf):
    if False:
        print('Hello World!')
    check_ocrmypdf(resources / 'crom.png', outpdf, '--image-dpi', '200', '--plugin', 'tests/plugins/tesseract_noop.py')

def test_no_dpi_info(caplog, baiona, outdir, no_outpdf):
    if False:
        return 10
    im = baiona
    assert 'dpi' not in im.info
    input_image = outdir / 'baiona_no_dpi.png'
    im.save(input_image)
    rc = run_ocrmypdf_api(input_image, no_outpdf)
    assert rc == ocrmypdf.ExitCode.input_file
    assert '--image-dpi' in caplog.text

def test_dpi_not_credible(caplog, baiona, outdir, no_outpdf):
    if False:
        print('Hello World!')
    im = baiona
    assert 'dpi' not in im.info
    input_image = outdir / 'baiona_no_dpi.png'
    im.save(input_image, dpi=(30, 30))
    rc = run_ocrmypdf_api(input_image, no_outpdf)
    assert rc == ocrmypdf.ExitCode.input_file
    assert 'not credible' in caplog.text

def test_cmyk_no_icc(caplog, resources, no_outpdf):
    if False:
        i = 10
        return i + 15
    rc = run_ocrmypdf_api(resources / 'baiona_cmyk.jpg', no_outpdf)
    assert rc == ocrmypdf.ExitCode.input_file
    assert 'no ICC profile' in caplog.text

def test_img2pdf_fails(resources, no_outpdf):
    if False:
        print('Hello World!')
    with patch('ocrmypdf._pipeline.img2pdf.convert', side_effect=img2pdf.ImageOpenError()) as mock:
        rc = run_ocrmypdf_api(resources / 'baiona_gray.png', no_outpdf, '--image-dpi', '200')
        assert rc == ocrmypdf.ExitCode.input_file
        mock.assert_called()

@pytest.mark.xfail(reason='remove background disabled')
def test_jpeg_in_jpeg_out(resources, outpdf):
    if False:
        while True:
            i = 10
    check_ocrmypdf(resources / 'baiona_color.jpg', outpdf, '--image-dpi', '100', '--output-type', 'pdf', '--remove-background', '--plugin', 'tests/plugins/tesseract_noop.py')
    with pikepdf.open(outpdf) as pdf:
        assert next(iter(pdf.pages[0].images.values())).Filter == pikepdf.Name.DCTDecode