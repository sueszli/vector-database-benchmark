"""Test the pypdf._merger module."""
import sys
from io import BytesIO
from pathlib import Path
import pytest
import pypdf
from pypdf import PdfMerger, PdfReader, PdfWriter
from pypdf.errors import DeprecationError
from pypdf.generic import Destination, Fit
from . import get_data_from_url
TESTS_ROOT = Path(__file__).parent.resolve()
PROJECT_ROOT = TESTS_ROOT.parent
RESOURCE_ROOT = PROJECT_ROOT / 'resources'
sys.path.append(str(PROJECT_ROOT))

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def merger_operate(merger):
    if False:
        i = 10
        return i + 15
    pdf_path = RESOURCE_ROOT / 'crazyones.pdf'
    outline = RESOURCE_ROOT / 'pdflatex-outline.pdf'
    pdf_forms = RESOURCE_ROOT / 'pdflatex-forms.pdf'
    pdf_pw = RESOURCE_ROOT / 'libreoffice-writer-password.pdf'
    merger.append(pdf_path)
    merger.append(outline)
    merger.append(pdf_path, pages=pypdf.pagerange.PageRange(slice(0, 0)))
    merger.append(pdf_forms)
    merger.merge(0, pdf_path, import_outline=False)
    with pytest.raises(NotImplementedError) as exc:
        with open(pdf_path, 'rb') as fp:
            data = fp.read()
        merger.append(data)
    assert exc.value.args[0].startswith('PdfMerger.merge requires an object that PdfReader can parse. Typically, that is a Path')
    reader = pypdf.PdfReader(pdf_pw)
    reader.decrypt('openpassword')
    merger.append(reader)
    r = pypdf.PdfReader(pdf_path)
    merger.append(r, outline_item='foo', pages=list(range(len(r.pages))))
    with open(pdf_path, 'rb') as fh:
        merger.append(fh)
    merger.write(BytesIO())
    outline_item = merger.add_outline_item('An outline item', 0)
    oi2 = merger.add_outline_item('deeper', 0, parent=outline_item, italic=True, bold=True)
    merger.add_outline_item("Let's see", 2, oi2, (255, 255, 0), True, True, Fit.fit_box_vertically(left=12))
    merger.add_outline_item('The XYZ fit', 0, outline_item, (255, 0, 15), True, True, Fit.xyz(left=10, top=20, zoom=3))
    merger.add_outline_item('The FitH fit', 0, outline_item, (255, 0, 15), True, True, Fit.fit_horizontally(top=10))
    merger.add_outline_item('The FitV fit', 0, outline_item, (255, 0, 15), True, True, Fit.fit_vertically(left=10))
    merger.add_outline_item('The FitR fit', 0, outline_item, (255, 0, 15), True, True, Fit.fit_rectangle(left=10, bottom=20, right=30, top=40))
    merger.add_outline_item('The FitB fit', 0, outline_item, (255, 0, 15), True, True, Fit.fit_box())
    merger.add_outline_item('The FitBH fit', 0, outline_item, (255, 0, 15), True, True, Fit.fit_box_horizontally(top=10))
    merger.add_outline_item('The FitBV fit', 0, outline_item, (255, 0, 15), True, True, Fit.fit_box_vertically(left=10))
    found_oi = merger.find_outline_item('nothing here')
    assert found_oi is None
    found_oi = merger.find_outline_item('foo')
    assert found_oi == [9]
    merger.add_metadata({'/Author': 'Martin Thoma'})
    merger.add_named_destination('/Title', 0)
    merger.set_page_layout('/SinglePage')
    merger.set_page_mode('/UseThumbs')

def check_outline(tmp_path):
    if False:
        return 10
    reader = pypdf.PdfReader(tmp_path)
    assert [el.title for el in reader.outline if isinstance(el, Destination)] == ['Foo', 'Bar', 'Baz', 'Foo', 'Bar', 'Baz', 'Foo', 'Bar', 'Baz', 'foo', 'An outline item']
tmp_filename = 'dont_commit_merged.pdf'

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_merger_operations_by_traditional_usage(tmp_path):
    if False:
        i = 10
        return i + 15
    merger = PdfMerger()
    merger_operate(merger)
    path = tmp_path / tmp_filename
    merger.write(path)
    merger.close()
    check_outline(path)

def test_merger_operations_by_traditional_usage_with_writer(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    merger = PdfWriter()
    merger_operate(merger)
    path = tmp_path / tmp_filename
    merger.write(path)
    merger.close()
    check_outline(path)

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_merger_operations_by_semi_traditional_usage(tmp_path):
    if False:
        print('Hello World!')
    path = tmp_path / tmp_filename
    with PdfMerger() as merger:
        merger_operate(merger)
        merger.write(path)
    assert Path(path).is_file()
    check_outline(path)

def test_merger_operations_by_semi_traditional_usage_with_writer(tmp_path):
    if False:
        while True:
            i = 10
    path = tmp_path / tmp_filename
    with PdfWriter() as merger:
        merger_operate(merger)
        merger.write(path)
    assert Path(path).is_file()
    check_outline(path)

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_merger_operation_by_new_usage(tmp_path):
    if False:
        return 10
    path = tmp_path / tmp_filename
    with PdfMerger(fileobj=path) as merger:
        merger_operate(merger)
    assert Path(path).is_file()
    check_outline(path)

def test_merger_operation_by_new_usage_with_writer(tmp_path):
    if False:
        print('Hello World!')
    path = tmp_path / tmp_filename
    with PdfWriter(fileobj=path) as merger:
        merger_operate(merger)
    assert Path(path).is_file()
    check_outline(path)

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_merge_page_exception():
    if False:
        print('Hello World!')
    merger = pypdf.PdfMerger()
    pdf_path = RESOURCE_ROOT / 'crazyones.pdf'
    with pytest.raises(TypeError) as exc:
        merger.merge(0, pdf_path, pages='a:b')
    assert exc.value.args[0] == '"pages" must be a tuple of (start, stop[, step])'
    merger.close()

def test_merge_page_exception_with_writer():
    if False:
        print('Hello World!')
    merger = pypdf.PdfWriter()
    pdf_path = RESOURCE_ROOT / 'crazyones.pdf'
    with pytest.raises(TypeError) as exc:
        merger.merge(0, pdf_path, pages='a:b')
    assert exc.value.args[0] == '"pages" must be a tuple of (start, stop[, step]) or a list'
    merger.close()

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_merge_page_tuple():
    if False:
        print('Hello World!')
    merger = pypdf.PdfMerger()
    pdf_path = RESOURCE_ROOT / 'crazyones.pdf'
    merger.merge(0, pdf_path, pages=(0, 1))
    merger.close()

def test_merge_page_tuple_with_writer():
    if False:
        print('Hello World!')
    merger = pypdf.PdfWriter()
    pdf_path = RESOURCE_ROOT / 'crazyones.pdf'
    merger.merge(0, pdf_path, pages=(0, 1))
    merger.close()

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_merge_write_closed_fh():
    if False:
        return 10
    merger = pypdf.PdfMerger()
    pdf_path = RESOURCE_ROOT / 'crazyones.pdf'
    merger.append(pdf_path)
    err_closed = 'close() was called and thus the writer cannot be used anymore'
    merger.close()
    with pytest.raises(RuntimeError) as exc:
        merger.write('test_merge_write_closed_fh.pdf')
    assert exc.value.args[0] == err_closed
    with pytest.raises(RuntimeError) as exc:
        merger.add_metadata({'author': 'Martin Thoma'})
    assert exc.value.args[0] == err_closed
    with pytest.raises(RuntimeError) as exc:
        merger.set_page_layout('/SinglePage')
    assert exc.value.args[0] == err_closed
    with pytest.raises(RuntimeError) as exc:
        merger.set_page_mode('/UseNone')
    assert exc.value.args[0] == err_closed
    with pytest.raises(RuntimeError) as exc:
        merger._write_outline()
    assert exc.value.args[0] == err_closed
    with pytest.raises(RuntimeError) as exc:
        merger.add_outline_item('An outline item', 0)
    assert exc.value.args[0] == err_closed
    with pytest.raises(RuntimeError) as exc:
        merger._write_dests()
    assert exc.value.args[0] == err_closed

def test_merge_write_closed_fh_with_writer(pdf_file_path):
    if False:
        for i in range(10):
            print('nop')
    merger = pypdf.PdfWriter()
    pdf_path = RESOURCE_ROOT / 'crazyones.pdf'
    merger.append(pdf_path)
    merger.close()
    merger.write(pdf_file_path)
    merger.add_metadata({'author': 'Martin Thoma'})
    merger.set_page_layout('/SinglePage')
    merger.set_page_mode('/UseNone')
    merger.add_outline_item('An outline item', 0)

@pytest.mark.enable_socket()
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_trim_outline_list(pdf_file_path):
    if False:
        while True:
            i = 10
    url = 'https://corpora.tika.apache.org/base/docs/govdocs1/995/995175.pdf'
    name = 'tika-995175.pdf'
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    merger = PdfMerger()
    merger.append(reader)
    merger.write(pdf_file_path)
    merger.close()

@pytest.mark.enable_socket()
def test_trim_outline_list_with_writer(pdf_file_path):
    if False:
        return 10
    url = 'https://corpora.tika.apache.org/base/docs/govdocs1/995/995175.pdf'
    name = 'tika-995175.pdf'
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    merger = PdfWriter()
    merger.append(reader)
    merger.write(pdf_file_path)
    merger.close()

@pytest.mark.enable_socket()
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_zoom(pdf_file_path):
    if False:
        print('Hello World!')
    url = 'https://corpora.tika.apache.org/base/docs/govdocs1/994/994759.pdf'
    name = 'tika-994759.pdf'
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    merger = PdfMerger()
    merger.append(reader)
    merger.write(pdf_file_path)
    merger.close()

@pytest.mark.enable_socket()
def test_zoom_with_writer(pdf_file_path):
    if False:
        i = 10
        return i + 15
    url = 'https://corpora.tika.apache.org/base/docs/govdocs1/994/994759.pdf'
    name = 'tika-994759.pdf'
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    merger = PdfWriter()
    merger.append(reader)
    merger.write(pdf_file_path)
    merger.close()

@pytest.mark.enable_socket()
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_zoom_xyz_no_left(pdf_file_path):
    if False:
        for i in range(10):
            print('nop')
    url = 'https://corpora.tika.apache.org/base/docs/govdocs1/933/933322.pdf'
    name = 'tika-933322.pdf'
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    merger = PdfMerger()
    merger.append(reader)
    merger.write(pdf_file_path)
    merger.close()

@pytest.mark.enable_socket()
def test_zoom_xyz_no_left_with_writer(pdf_file_path):
    if False:
        return 10
    url = 'https://corpora.tika.apache.org/base/docs/govdocs1/933/933322.pdf'
    name = 'tika-933322.pdf'
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    merger = PdfWriter()
    merger.append(reader)
    merger.write(pdf_file_path)
    merger.close()

@pytest.mark.enable_socket()
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_outline_item(pdf_file_path):
    if False:
        for i in range(10):
            print('nop')
    url = 'https://corpora.tika.apache.org/base/docs/govdocs1/997/997511.pdf'
    name = 'tika-997511.pdf'
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    merger = PdfMerger()
    merger.append(reader)
    merger.write(pdf_file_path)
    merger.close()

@pytest.mark.enable_socket()
@pytest.mark.slow()
def test_outline_item_with_writer(pdf_file_path):
    if False:
        print('Hello World!')
    url = 'https://corpora.tika.apache.org/base/docs/govdocs1/997/997511.pdf'
    name = 'tika-997511.pdf'
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    merger = PdfWriter()
    merger.append(reader)
    merger.write(pdf_file_path)
    merger.close()

@pytest.mark.enable_socket()
@pytest.mark.slow()
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_trim_outline(pdf_file_path):
    if False:
        while True:
            i = 10
    url = 'https://corpora.tika.apache.org/base/docs/govdocs1/982/982336.pdf'
    name = 'tika-982336.pdf'
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    merger = PdfMerger()
    merger.append(reader)
    merger.write(pdf_file_path)
    merger.close()

@pytest.mark.enable_socket()
@pytest.mark.slow()
def test_trim_outline_with_writer(pdf_file_path):
    if False:
        print('Hello World!')
    url = 'https://corpora.tika.apache.org/base/docs/govdocs1/982/982336.pdf'
    name = 'tika-982336.pdf'
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    merger = PdfWriter()
    merger.append(reader)
    merger.write(pdf_file_path)
    merger.close()

@pytest.mark.enable_socket()
@pytest.mark.slow()
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test1(pdf_file_path):
    if False:
        print('Hello World!')
    url = 'https://corpora.tika.apache.org/base/docs/govdocs1/923/923621.pdf'
    name = 'tika-923621.pdf'
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    merger = PdfMerger()
    merger.append(reader)
    merger.write(pdf_file_path)
    merger.close()

@pytest.mark.enable_socket()
@pytest.mark.slow()
def test1_with_writer(pdf_file_path):
    if False:
        i = 10
        return i + 15
    url = 'https://corpora.tika.apache.org/base/docs/govdocs1/923/923621.pdf'
    name = 'tika-923621.pdf'
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    merger = PdfWriter()
    merger.append(reader)
    merger.write(pdf_file_path)
    merger.close()

@pytest.mark.enable_socket()
@pytest.mark.slow()
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_sweep_recursion1(pdf_file_path):
    if False:
        return 10
    url = 'https://corpora.tika.apache.org/base/docs/govdocs1/924/924546.pdf'
    name = 'tika-924546.pdf'
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    merger = PdfMerger()
    merger.append(reader)
    merger.write(pdf_file_path)
    merger.close()
    reader2 = PdfReader(pdf_file_path)
    reader2.pages

@pytest.mark.enable_socket()
@pytest.mark.slow()
def test_sweep_recursion1_with_writer(pdf_file_path):
    if False:
        print('Hello World!')
    url = 'https://corpora.tika.apache.org/base/docs/govdocs1/924/924546.pdf'
    name = 'tika-924546.pdf'
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    merger = PdfWriter()
    merger.append(reader)
    merger.write(pdf_file_path)
    merger.close()
    reader2 = PdfReader(pdf_file_path)
    reader2.pages

@pytest.mark.enable_socket()
@pytest.mark.slow()
@pytest.mark.parametrize(('url', 'name'), [('https://corpora.tika.apache.org/base/docs/govdocs1/924/924794.pdf', 'tika-924794.pdf'), ('https://corpora.tika.apache.org/base/docs/govdocs1/924/924546.pdf', 'tika-924546.pdf')])
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_sweep_recursion2(url, name, pdf_file_path):
    if False:
        for i in range(10):
            print('nop')
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    merger = PdfMerger()
    merger.append(reader)
    merger.write(pdf_file_path)
    merger.close()
    reader2 = PdfReader(pdf_file_path)
    reader2.pages

@pytest.mark.enable_socket()
@pytest.mark.slow()
@pytest.mark.parametrize(('url', 'name'), [('https://corpora.tika.apache.org/base/docs/govdocs1/924/924794.pdf', 'tika-924794.pdf'), ('https://corpora.tika.apache.org/base/docs/govdocs1/924/924546.pdf', 'tika-924546.pdf')])
def test_sweep_recursion2_with_writer(url, name, pdf_file_path):
    if False:
        for i in range(10):
            print('nop')
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    merger = PdfWriter()
    merger.append(reader)
    merger.write(pdf_file_path)
    merger.close()
    reader2 = PdfReader(pdf_file_path)
    reader2.pages

@pytest.mark.enable_socket()
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_sweep_indirect_list_newobj_is_none(caplog, pdf_file_path):
    if False:
        return 10
    url = 'https://corpora.tika.apache.org/base/docs/govdocs1/906/906769.pdf'
    name = 'tika-906769.pdf'
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    merger = PdfMerger()
    merger.append(reader)
    merger.write(pdf_file_path)
    merger.close()
    reader2 = PdfReader(pdf_file_path)
    reader2.pages

@pytest.mark.enable_socket()
def test_sweep_indirect_list_newobj_is_none_with_writer(caplog, pdf_file_path):
    if False:
        print('Hello World!')
    url = 'https://corpora.tika.apache.org/base/docs/govdocs1/906/906769.pdf'
    name = 'tika-906769.pdf'
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    merger = PdfWriter()
    merger.append(reader)
    merger.write(pdf_file_path)
    merger.close()
    reader2 = PdfReader(pdf_file_path)
    reader2.pages

@pytest.mark.enable_socket()
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_iss1145():
    if False:
        return 10
    url = 'https://github.com/py-pdf/pypdf/files/9164743/file-0.pdf'
    name = 'iss1145.pdf'
    merger = PdfMerger()
    merger.append(PdfReader(BytesIO(get_data_from_url(url, name=name))))
    merger.close()

@pytest.mark.enable_socket()
def test_iss1145_with_writer():
    if False:
        i = 10
        return i + 15
    url = 'https://github.com/py-pdf/pypdf/files/9164743/file-0.pdf'
    name = 'iss1145.pdf'
    merger = PdfWriter()
    merger.append(PdfReader(BytesIO(get_data_from_url(url, name=name))))
    merger.close()

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_deprecation_bookmark_decorator_deprecationexcp():
    if False:
        print('Hello World!')
    reader = PdfReader(RESOURCE_ROOT / 'outlines-with-invalid-destinations.pdf')
    merger = PdfMerger()
    with pytest.raises(DeprecationError, match='import_bookmarks is deprecated as an argument. Use import_outline instead'):
        merger.merge(0, reader, import_bookmarks=True)

def test_deprecation_bookmark_decorator_deprecationexcp_with_writer():
    if False:
        while True:
            i = 10
    reader = PdfReader(RESOURCE_ROOT / 'outlines-with-invalid-destinations.pdf')
    merger = PdfWriter()
    with pytest.raises(DeprecationError, match='import_bookmarks is deprecated as an argument. Use import_outline instead'):
        merger.merge(0, reader, import_bookmarks=True)

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_deprecation_bookmark_decorator_output():
    if False:
        print('Hello World!')
    reader = PdfReader(RESOURCE_ROOT / 'outlines-with-invalid-destinations.pdf')
    merger = PdfMerger()
    with pytest.raises(DeprecationError):
        merger.merge(0, reader, import_bookmarks=True)

def test_deprecation_bookmark_decorator_output_with_writer():
    if False:
        return 10
    reader = PdfReader(RESOURCE_ROOT / 'outlines-with-invalid-destinations.pdf')
    merger = PdfWriter()
    with pytest.raises(DeprecationError):
        merger.merge(0, reader, import_bookmarks=True)

@pytest.mark.enable_socket()
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_iss1344(caplog):
    if False:
        for i in range(10):
            print('nop')
    url = 'https://github.com/py-pdf/pypdf/files/9549001/input.pdf'
    name = 'iss1344.pdf'
    m = PdfMerger()
    m.append(PdfReader(BytesIO(get_data_from_url(url, name=name))))
    b = BytesIO()
    m.write(b)
    r = PdfReader(b)
    p = r.pages[0]
    assert '/DIJMAC+Arial Black' in p._debug_for_extract()
    assert 'adresse où le malade peut être visité' in p.extract_text()
    assert r.threads is None

@pytest.mark.enable_socket()
def test_iss1344_with_writer(caplog):
    if False:
        for i in range(10):
            print('nop')
    url = 'https://github.com/py-pdf/pypdf/files/9549001/input.pdf'
    name = 'iss1344.pdf'
    m = PdfWriter()
    m.append(PdfReader(BytesIO(get_data_from_url(url, name=name))))
    b = BytesIO()
    m.write(b)
    p = PdfReader(b).pages[0]
    assert '/DIJMAC+Arial Black' in p._debug_for_extract()
    assert 'adresse où le malade peut être visité' in p.extract_text()

@pytest.mark.enable_socket()
def test_articles_with_writer(caplog):
    if False:
        i = 10
        return i + 15
    url = 'https://corpora.tika.apache.org/base/docs/govdocs1/924/924666.pdf'
    name = '924666.pdf'
    m = PdfWriter()
    m.append(PdfReader(BytesIO(get_data_from_url(url, name=name))), (2, 10))
    b = BytesIO()
    m.write(b)
    r = PdfReader(b)
    assert len(r.threads) == 4
    assert r.threads[0].get_object()['/F']['/P'] == r.pages[0]

def test_deprecate_pdfmerger():
    if False:
        i = 10
        return i + 15
    with pytest.warns(DeprecationWarning), PdfMerger() as merger:
        merger.append(RESOURCE_ROOT / 'crazyones.pdf')