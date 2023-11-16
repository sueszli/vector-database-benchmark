"""Test images util."""
import pytest
from sphinx.util.images import get_image_extension, get_image_size, guess_mimetype, parse_data_uri
GIF_FILENAME = 'img.gif'
PNG_FILENAME = 'img.png'
PDF_FILENAME = 'img.pdf'
TXT_FILENAME = 'index.txt'

def test_get_image_size(rootdir):
    if False:
        print('Hello World!')
    assert get_image_size(rootdir / 'test-root' / GIF_FILENAME) == (200, 181)
    assert get_image_size(rootdir / 'test-root' / PNG_FILENAME) == (200, 181)
    assert get_image_size(rootdir / 'test-root' / PDF_FILENAME) is None
    assert get_image_size(rootdir / 'test-root' / TXT_FILENAME) is None

@pytest.mark.filterwarnings('ignore:The content argument')
def test_guess_mimetype():
    if False:
        i = 10
        return i + 15
    assert guess_mimetype('img.png') == 'image/png'
    assert guess_mimetype('img.jpg') == 'image/jpeg'
    assert guess_mimetype('img.txt') is None
    assert guess_mimetype('img.txt', default='text/plain') == 'text/plain'
    assert guess_mimetype('no_extension') is None
    assert guess_mimetype('IMG.PNG') == 'image/png'
    assert guess_mimetype('img.png', 'text/plain') == 'image/png'
    assert guess_mimetype('no_extension', 'text/plain') == 'text/plain'

def test_get_image_extension():
    if False:
        for i in range(10):
            print('nop')
    assert get_image_extension('image/png') == '.png'
    assert get_image_extension('image/jpeg') == '.jpg'
    assert get_image_extension('image/svg+xml') == '.svg'
    assert get_image_extension('text/plain') is None

def test_parse_data_uri():
    if False:
        for i in range(10):
            print('nop')
    uri = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=='
    image = parse_data_uri(uri)
    assert image is not None
    assert image.mimetype == 'image/png'
    assert image.charset == 'US-ASCII'
    uri = 'data:charset=utf-8,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=='
    image = parse_data_uri(uri)
    assert image is not None
    assert image.mimetype == 'text/plain'
    assert image.charset == 'utf-8'
    uri = 'image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=='
    image = parse_data_uri(uri)
    assert image is None
    uri = 'data:iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=='
    with pytest.raises(ValueError, match='not enough values to unpack \\(expected 2, got 1\\)'):
        parse_data_uri(uri)