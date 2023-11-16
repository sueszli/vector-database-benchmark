from io import BytesIO
import pytest
import logging
from matplotlib import _afm
from matplotlib import font_manager as fm
AFM_TEST_DATA = b'StartFontMetrics 2.0\nComment Comments are ignored.\nComment Creation Date:Mon Nov 13 12:34:11 GMT 2017\nFontName MyFont-Bold\nEncodingScheme FontSpecific\nFullName My Font Bold\nFamilyName Test Fonts\nWeight Bold\nItalicAngle 0.0\nIsFixedPitch false\nUnderlinePosition -100\nUnderlineThickness 56,789\nVersion 001.000\nNotice Copyright \xa9 2017 No one.\nFontBBox 0 -321 1234 369\nStartCharMetrics 3\nC 0 ; WX 250 ; N space ; B 0 0 0 0 ;\nC 42 ; WX 1141 ; N foo ; B 40 60 800 360 ;\nC 99 ; WX 583 ; N bar ; B 40 -10 543 210 ;\nEndCharMetrics\nEndFontMetrics\n'

def test_nonascii_str():
    if False:
        for i in range(10):
            print('nop')
    inp_str = 'привет'
    byte_str = inp_str.encode('utf8')
    ret = _afm._to_str(byte_str)
    assert ret == inp_str

def test_parse_header():
    if False:
        return 10
    fh = BytesIO(AFM_TEST_DATA)
    header = _afm._parse_header(fh)
    assert header == {b'StartFontMetrics': 2.0, b'FontName': 'MyFont-Bold', b'EncodingScheme': 'FontSpecific', b'FullName': 'My Font Bold', b'FamilyName': 'Test Fonts', b'Weight': 'Bold', b'ItalicAngle': 0.0, b'IsFixedPitch': False, b'UnderlinePosition': -100, b'UnderlineThickness': 56.789, b'Version': '001.000', b'Notice': b'Copyright \xa9 2017 No one.', b'FontBBox': [0, -321, 1234, 369], b'StartCharMetrics': 3}

def test_parse_char_metrics():
    if False:
        i = 10
        return i + 15
    fh = BytesIO(AFM_TEST_DATA)
    _afm._parse_header(fh)
    metrics = _afm._parse_char_metrics(fh)
    assert metrics == ({0: (250.0, 'space', [0, 0, 0, 0]), 42: (1141.0, 'foo', [40, 60, 800, 360]), 99: (583.0, 'bar', [40, -10, 543, 210])}, {'space': (250.0, 'space', [0, 0, 0, 0]), 'foo': (1141.0, 'foo', [40, 60, 800, 360]), 'bar': (583.0, 'bar', [40, -10, 543, 210])})

def test_get_familyname_guessed():
    if False:
        i = 10
        return i + 15
    fh = BytesIO(AFM_TEST_DATA)
    font = _afm.AFM(fh)
    del font._header[b'FamilyName']
    assert font.get_familyname() == 'My Font'

def test_font_manager_weight_normalization():
    if False:
        print('Hello World!')
    font = _afm.AFM(BytesIO(AFM_TEST_DATA.replace(b'Weight Bold\n', b'Weight Custom\n')))
    assert fm.afmFontProperty('', font).weight == 'normal'

@pytest.mark.parametrize('afm_data', [b'nope\nreally nope', b'StartFontMetrics 2.0\nComment Comments are ignored.\nComment Creation Date:Mon Nov 13 12:34:11 GMT 2017\nFontName MyFont-Bold\nEncodingScheme FontSpecific'])
def test_bad_afm(afm_data):
    if False:
        for i in range(10):
            print('nop')
    fh = BytesIO(afm_data)
    with pytest.raises(RuntimeError):
        _afm._parse_header(fh)

@pytest.mark.parametrize('afm_data', [b'StartFontMetrics 2.0\nComment Comments are ignored.\nComment Creation Date:Mon Nov 13 12:34:11 GMT 2017\nAardvark bob\nFontName MyFont-Bold\nEncodingScheme FontSpecific\nStartCharMetrics 3', b'StartFontMetrics 2.0\nComment Comments are ignored.\nComment Creation Date:Mon Nov 13 12:34:11 GMT 2017\nItalicAngle zero degrees\nFontName MyFont-Bold\nEncodingScheme FontSpecific\nStartCharMetrics 3'])
def test_malformed_header(afm_data, caplog):
    if False:
        i = 10
        return i + 15
    fh = BytesIO(afm_data)
    with caplog.at_level(logging.ERROR):
        _afm._parse_header(fh)
    assert len(caplog.records) == 1