import io
import os
import os.path
import tempfile
import time
import pytest
from PIL import Image, PdfParser, features
from .helper import hopper, mark_if_feature_version, skip_unless_feature

def helper_save_as_pdf(tmp_path, mode, **kwargs):
    if False:
        i = 10
        return i + 15
    im = hopper(mode)
    outfile = str(tmp_path / ('temp_' + mode + '.pdf'))
    im.save(outfile, **kwargs)
    assert os.path.isfile(outfile)
    assert os.path.getsize(outfile) > 0
    with PdfParser.PdfParser(outfile) as pdf:
        if kwargs.get('append_images', False) or kwargs.get('append', False):
            assert len(pdf.pages) > 1
        else:
            assert len(pdf.pages) > 0
    with open(outfile, 'rb') as fp:
        contents = fp.read()
    size = tuple((float(d) for d in contents.split(b'/MediaBox [ 0 0 ')[1].split(b']')[0].split()))
    assert im.size == size
    return outfile

@pytest.mark.parametrize('mode', ('L', 'P', 'RGB', 'CMYK'))
def test_save(tmp_path, mode):
    if False:
        for i in range(10):
            print('nop')
    helper_save_as_pdf(tmp_path, mode)

@skip_unless_feature('jpg_2000')
@pytest.mark.parametrize('mode', ('LA', 'RGBA'))
def test_save_alpha(tmp_path, mode):
    if False:
        print('Hello World!')
    helper_save_as_pdf(tmp_path, mode)

def test_p_alpha(tmp_path):
    if False:
        i = 10
        return i + 15
    outfile = str(tmp_path / 'temp.pdf')
    with Image.open('Tests/images/pil123p.png') as im:
        assert im.mode == 'P'
        assert isinstance(im.info['transparency'], bytes)
        im.save(outfile)
    with open(outfile, 'rb') as fp:
        contents = fp.read()
    assert b'\n/SMask ' in contents

def test_monochrome(tmp_path):
    if False:
        while True:
            i = 10
    mode = '1'
    outfile = helper_save_as_pdf(tmp_path, mode)
    assert os.path.getsize(outfile) < (5000 if features.check('libtiff') else 15000)

def test_unsupported_mode(tmp_path):
    if False:
        print('Hello World!')
    im = hopper('PA')
    outfile = str(tmp_path / 'temp_PA.pdf')
    with pytest.raises(ValueError):
        im.save(outfile)

def test_resolution(tmp_path):
    if False:
        while True:
            i = 10
    im = hopper()
    outfile = str(tmp_path / 'temp.pdf')
    im.save(outfile, resolution=150)
    with open(outfile, 'rb') as fp:
        contents = fp.read()
    size = tuple((float(d) for d in contents.split(b'stream\nq ')[1].split(b' 0 0 cm')[0].split(b' 0 0 ')))
    assert size == (61.44, 61.44)
    size = tuple((float(d) for d in contents.split(b'/MediaBox [ 0 0 ')[1].split(b']')[0].split()))
    assert size == (61.44, 61.44)

@pytest.mark.parametrize('params', ({'dpi': (75, 150)}, {'dpi': (75, 150), 'resolution': 200}))
def test_dpi(params, tmp_path):
    if False:
        while True:
            i = 10
    im = hopper()
    outfile = str(tmp_path / 'temp.pdf')
    im.save(outfile, **params)
    with open(outfile, 'rb') as fp:
        contents = fp.read()
    size = tuple((float(d) for d in contents.split(b'stream\nq ')[1].split(b' 0 0 cm')[0].split(b' 0 0 ')))
    assert size == (122.88, 61.44)
    size = tuple((float(d) for d in contents.split(b'/MediaBox [ 0 0 ')[1].split(b']')[0].split()))
    assert size == (122.88, 61.44)

@mark_if_feature_version(pytest.mark.valgrind_known_error, 'libjpeg_turbo', '2.0', reason='Known Failing')
def test_save_all(tmp_path):
    if False:
        i = 10
        return i + 15
    helper_save_as_pdf(tmp_path, 'RGB', save_all=True)
    with Image.open('Tests/images/dispose_bgnd.gif') as im:
        outfile = str(tmp_path / 'temp.pdf')
        im.save(outfile, save_all=True)
        assert os.path.isfile(outfile)
        assert os.path.getsize(outfile) > 0
        ims = [hopper()]
        im.copy().save(outfile, save_all=True, append_images=ims)
        assert os.path.isfile(outfile)
        assert os.path.getsize(outfile) > 0

        def im_generator(ims):
            if False:
                return 10
            yield from ims
        im.save(outfile, save_all=True, append_images=im_generator(ims))
    assert os.path.isfile(outfile)
    assert os.path.getsize(outfile) > 0
    with Image.open('Tests/images/flower.jpg') as jpeg:
        jpeg.save(outfile, save_all=True, append_images=[jpeg.copy()])
    assert os.path.isfile(outfile)
    assert os.path.getsize(outfile) > 0

def test_multiframe_normal_save(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    with Image.open('Tests/images/dispose_bgnd.gif') as im:
        outfile = str(tmp_path / 'temp.pdf')
        im.save(outfile)
    assert os.path.isfile(outfile)
    assert os.path.getsize(outfile) > 0

def test_pdf_open(tmp_path):
    if False:
        return 10
    with pytest.raises(PdfParser.PdfFormatError):
        PdfParser.PdfParser(buf=bytearray(65536))
    with PdfParser.PdfParser() as empty_pdf:
        assert len(empty_pdf.pages) == 0
        assert len(empty_pdf.info) == 0
        assert not empty_pdf.should_close_buf
        assert not empty_pdf.should_close_file
    pdf_filename = helper_save_as_pdf(tmp_path, 'RGB')
    with PdfParser.PdfParser(filename=pdf_filename) as hopper_pdf:
        assert len(hopper_pdf.pages) == 1
        assert hopper_pdf.should_close_buf
        assert hopper_pdf.should_close_file
    with open(pdf_filename, 'rb') as f:
        content = b'xyzzy' + f.read()
    with PdfParser.PdfParser(buf=content, start_offset=5) as hopper_pdf:
        assert len(hopper_pdf.pages) == 1
        assert not hopper_pdf.should_close_buf
        assert not hopper_pdf.should_close_file
    with open(pdf_filename, 'rb') as f:
        with PdfParser.PdfParser(f=f) as hopper_pdf:
            assert len(hopper_pdf.pages) == 1
            assert hopper_pdf.should_close_buf
            assert not hopper_pdf.should_close_file

def test_pdf_append_fails_on_nonexistent_file():
    if False:
        for i in range(10):
            print('nop')
    im = hopper('RGB')
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(OSError):
            im.save(os.path.join(temp_dir, 'nonexistent.pdf'), append=True)

def check_pdf_pages_consistency(pdf):
    if False:
        for i in range(10):
            print('nop')
    pages_info = pdf.read_indirect(pdf.pages_ref)
    assert b'Parent' not in pages_info
    assert b'Kids' in pages_info
    kids_not_used = pages_info[b'Kids']
    for page_ref in pdf.pages:
        while True:
            if page_ref in kids_not_used:
                kids_not_used.remove(page_ref)
            page_info = pdf.read_indirect(page_ref)
            assert b'Parent' in page_info
            page_ref = page_info[b'Parent']
            if page_ref == pdf.pages_ref:
                break
        assert pdf.pages_ref == page_info[b'Parent']
    assert kids_not_used == []

def test_pdf_append(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    pdf_filename = helper_save_as_pdf(tmp_path, 'RGB', producer='PdfParser')
    with PdfParser.PdfParser(pdf_filename, mode='r+b') as pdf:
        assert len(pdf.pages) == 1
        assert len(pdf.info) == 4
        assert pdf.info.Title == os.path.splitext(os.path.basename(pdf_filename))[0]
        assert pdf.info.Producer == 'PdfParser'
        assert b'CreationDate' in pdf.info
        assert b'ModDate' in pdf.info
        check_pdf_pages_consistency(pdf)
        pdf.info.Title = 'abc'
        pdf.info.Author = 'def'
        pdf.info.Subject = 'ghiꯍ'
        pdf.info.Keywords = 'qw)e\\r(ty'
        pdf.info.Creator = 'hopper()'
        pdf.start_writing()
        pdf.write_xref_and_trailer()
    with PdfParser.PdfParser(pdf_filename) as pdf:
        assert len(pdf.pages) == 1
        assert len(pdf.info) == 8
        assert pdf.info.Title == 'abc'
        assert b'CreationDate' in pdf.info
        assert b'ModDate' in pdf.info
        check_pdf_pages_consistency(pdf)
    mode_cmyk = hopper('CMYK')
    mode_p = hopper('P')
    mode_cmyk.save(pdf_filename, append=True, save_all=True, append_images=[mode_p])
    with PdfParser.PdfParser(pdf_filename) as pdf:
        assert len(pdf.pages) == 3
        assert len(pdf.info) == 8
        assert PdfParser.decode_text(pdf.info[b'Title']) == 'abc'
        assert pdf.info.Title == 'abc'
        assert pdf.info.Producer == 'PdfParser'
        assert pdf.info.Keywords == 'qw)e\\r(ty'
        assert pdf.info.Subject == 'ghiꯍ'
        assert b'CreationDate' in pdf.info
        assert b'ModDate' in pdf.info
        check_pdf_pages_consistency(pdf)

def test_pdf_info(tmp_path):
    if False:
        while True:
            i = 10
    pdf_filename = helper_save_as_pdf(tmp_path, 'RGB', title='title', author='author', subject='subject', keywords='keywords', creator='creator', producer='producer', creationDate=time.strptime('2000', '%Y'), modDate=time.strptime('2001', '%Y'))
    with PdfParser.PdfParser(pdf_filename) as pdf:
        assert len(pdf.info) == 8
        assert pdf.info.Title == 'title'
        assert pdf.info.Author == 'author'
        assert pdf.info.Subject == 'subject'
        assert pdf.info.Keywords == 'keywords'
        assert pdf.info.Creator == 'creator'
        assert pdf.info.Producer == 'producer'
        assert pdf.info.CreationDate == time.strptime('2000', '%Y')
        assert pdf.info.ModDate == time.strptime('2001', '%Y')
        check_pdf_pages_consistency(pdf)

def test_pdf_append_to_bytesio():
    if False:
        while True:
            i = 10
    im = hopper('RGB')
    f = io.BytesIO()
    im.save(f, format='PDF')
    initial_size = len(f.getvalue())
    assert initial_size > 0
    im = hopper('P')
    f = io.BytesIO(f.getvalue())
    im.save(f, format='PDF', append=True)
    assert len(f.getvalue()) > initial_size

@pytest.mark.timeout(1)
@pytest.mark.skipif('PILLOW_VALGRIND_TEST' in os.environ, reason='Valgrind is slower')
@pytest.mark.parametrize('newline', (b'\r', b'\n'))
def test_redos(newline):
    if False:
        return 10
    malicious = b' trailer<<>>' + newline * 3456
    with pytest.raises(PdfParser.PdfFormatError):
        PdfParser.PdfParser(buf=malicious)