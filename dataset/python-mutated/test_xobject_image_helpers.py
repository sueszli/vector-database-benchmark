"""Test the pypdf._xobj_image_helpers module."""
from io import BytesIO
import pytest
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from . import get_data_from_url

@pytest.mark.enable_socket()
def test_get_imagemode_recursion_depth():
    if False:
        for i in range(10):
            print('nop')
    'Avoid infinite recursion for nested color spaces.'
    url = 'https://github.com/py-pdf/pypdf/files/12814018/out1.pdf'
    name = 'issue2240.pdf'
    content = get_data_from_url(url, name=name)
    source = b'\n10 0 obj\n[ /DeviceN [ /HKS#2044#20K /Magenta /Yellow /Black ] 7 0 R 11 0 R 12 0 R ]\nendobj\n'
    target = b'\n10 0 obj\n[ /DeviceN [ /HKS#2044#20K /Magenta /Yellow /Black ] 10 0 R 11 0 R 12 0 R ]\nendobj\n'
    reader = PdfReader(BytesIO(content.replace(source, target)))
    with pytest.raises(PdfReadError, match='Color spaces nested too deep. If required, consider increasing MAX_IMAGE_MODE_NESTING_DEPTH.'):
        reader.pages[0].images[0]