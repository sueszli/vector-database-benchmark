import functools
import io
import os
import re
import sys
from contextlib import nullcontext
from io import BytesIO
from textwrap import dedent
import numpy as np
import pytest
from numpy import ma
from astropy.io import ascii
from astropy.io.ascii.core import FastOptionsError, InconsistentTableError, ParameterError
from astropy.io.ascii.fastbasic import FastBasic, FastCommentedHeader, FastCsv, FastNoHeader, FastRdb, FastTab
from astropy.table import MaskedColumn, Table
from astropy.tests.helper import CI
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.exceptions import AstropyWarning
from .common import assert_almost_equal, assert_equal, assert_true
StringIO = lambda x: BytesIO(x.encode('ascii'))

def assert_table_equal(t1, t2, check_meta=False, rtol=1e-15, atol=1e-300):
    if False:
        i = 10
        return i + 15
    '\n    Test equality of all columns in a table, with stricter tolerances for\n    float columns than the np.allclose default.\n    '
    assert_equal(len(t1), len(t2))
    assert_equal(t1.colnames, t2.colnames)
    if check_meta:
        assert_equal(t1.meta, t2.meta)
    for name in t1.colnames:
        if len(t1) != 0:
            assert_equal(t1[name].dtype.kind, t2[name].dtype.kind)
        if not isinstance(t1[name], MaskedColumn):
            for (i, el) in enumerate(t1[name]):
                try:
                    if not isinstance(el, str) and np.isnan(el):
                        assert_true(not isinstance(t2[name][i], str) and np.isnan(t2[name][i]))
                    elif isinstance(el, str):
                        assert_equal(el, t2[name][i])
                    else:
                        assert_almost_equal(el, t2[name][i], rtol=rtol, atol=atol)
                except (TypeError, NotImplementedError):
                    pass
_filename_counter = 0

def _read(tmp_path, table, reader_cls=None, format=None, parallel=False, check_meta=False, **kwargs):
    if False:
        print('Hello World!')
    global _filename_counter
    table += '\n'
    reader = reader_cls(**kwargs)
    t1 = reader.read(table)
    t2 = reader.read(StringIO(table))
    t3 = reader.read(table.splitlines())
    t4 = ascii.read(table, format=format, guess=False, **kwargs)
    t5 = ascii.read(table, format=format, guess=False, fast_reader=False, **kwargs)
    assert_table_equal(t1, t2, check_meta=check_meta)
    assert_table_equal(t2, t3, check_meta=check_meta)
    assert_table_equal(t3, t4, check_meta=check_meta)
    assert_table_equal(t4, t5, check_meta=check_meta)
    if parallel:
        if CI:
            pytest.xfail('Multiprocessing can sometimes fail on CI')
        t6 = ascii.read(table, format=format, guess=False, fast_reader={'parallel': True}, **kwargs)
        assert_table_equal(t1, t6, check_meta=check_meta)
    filename = tmp_path / f'table{_filename_counter}.txt'
    _filename_counter += 1
    with open(filename, 'wb') as f:
        f.write(table.encode('ascii'))
        f.flush()
    t7 = ascii.read(filename, format=format, guess=False, **kwargs)
    if parallel:
        t8 = ascii.read(filename, format=format, guess=False, fast_reader={'parallel': True}, **kwargs)
    assert_table_equal(t1, t7, check_meta=check_meta)
    if parallel:
        assert_table_equal(t1, t8, check_meta=check_meta)
    return t1

@pytest.fixture(scope='function')
def read_basic(tmp_path, request):
    if False:
        i = 10
        return i + 15
    return functools.partial(_read, tmp_path, reader_cls=FastBasic, format='basic')

@pytest.fixture(scope='function')
def read_csv(tmp_path, request):
    if False:
        print('Hello World!')
    return functools.partial(_read, tmp_path, reader_cls=FastCsv, format='csv')

@pytest.fixture(scope='function')
def read_tab(tmp_path, request):
    if False:
        i = 10
        return i + 15
    return functools.partial(_read, tmp_path, reader_cls=FastTab, format='tab')

@pytest.fixture(scope='function')
def read_commented_header(tmp_path, request):
    if False:
        while True:
            i = 10
    return functools.partial(_read, tmp_path, reader_cls=FastCommentedHeader, format='commented_header')

@pytest.fixture(scope='function')
def read_rdb(tmp_path, request):
    if False:
        for i in range(10):
            print('nop')
    return functools.partial(_read, tmp_path, reader_cls=FastRdb, format='rdb')

@pytest.fixture(scope='function')
def read_no_header(tmp_path, request):
    if False:
        return 10
    return functools.partial(_read, tmp_path, reader_cls=FastNoHeader, format='no_header')

@pytest.mark.parametrize('delimiter', [',', '\t', ' ', 'csv'])
@pytest.mark.parametrize('quotechar', ['"', "'"])
@pytest.mark.parametrize('fast', [False, True])
def test_embedded_newlines(delimiter, quotechar, fast):
    if False:
        for i in range(10):
            print('nop')
    'Test that embedded newlines are supported for io.ascii readers\n    and writers, both fast and Python readers.'
    dat = [['\t a ', ' b \n cd ', '\n'], [' 1\n ', '2 \n" \t 3\n4\n5', "1\n '2\n"], [' x,y \nz\t', '\t 12\n\t34\t ', '56\t\n']]
    dat = Table(dat, names=('a', 'b', 'c'))
    exp = {}
    for col in dat.itercols():
        vals = []
        for val in col:
            val = val.strip(' \t')
            if not fast:
                bits = val.splitlines(keepends=True)
                bits_out = []
                for bit in bits:
                    bit = re.sub('[ \\t]+(\\n?)$', '\\1', bit.strip(' \t'))
                    bits_out.append(bit)
                val = ''.join(bits_out)
            vals.append(val)
        exp[col.info.name] = vals
    exp = Table(exp)
    if delimiter == 'csv':
        format = 'csv'
        delimiter = ','
    else:
        format = 'basic'
    fh = io.StringIO()
    ascii.write(dat, fh, format=format, delimiter=delimiter, quotechar=quotechar, fast_writer=fast)
    text = fh.getvalue()
    dat_out = ascii.read(text, format=format, guess=False, delimiter=delimiter, quotechar=quotechar, fast_reader=fast)
    eq = dat_out.values_equal(exp)
    assert all((np.all(col) for col in eq.itercols()))

@pytest.mark.parametrize('parallel', [True, False])
def test_simple_data(parallel, read_basic):
    if False:
        print('Hello World!')
    '\n    Make sure the fast reader works with basic input data.\n    '
    table = read_basic('A B C\n1 2 3\n4 5 6', parallel=parallel)
    expected = Table([[1, 4], [2, 5], [3, 6]], names=('A', 'B', 'C'))
    assert_table_equal(table, expected)

def test_read_types():
    if False:
        return 10
    '\n    Make sure that the read() function takes filenames,\n    strings, and lists of strings in addition to file-like objects.\n    '
    t1 = ascii.read('a b c\n1 2 3\n4 5 6', format='fast_basic', guess=False)
    t2 = ascii.read(StringIO('a b c\n1 2 3\n4 5 6'), format='fast_basic', guess=False)
    t3 = ascii.read(['a b c', '1 2 3', '4 5 6'], format='fast_basic', guess=False)
    assert_table_equal(t1, t2)
    assert_table_equal(t2, t3)

@pytest.mark.parametrize('parallel', [True, False])
def test_supplied_names(parallel, read_basic):
    if False:
        i = 10
        return i + 15
    '\n    If passed as a parameter, names should replace any\n    column names found in the header.\n    '
    table = read_basic('A B C\n1 2 3\n4 5 6', names=('X', 'Y', 'Z'), parallel=parallel)
    expected = Table([[1, 4], [2, 5], [3, 6]], names=('X', 'Y', 'Z'))
    assert_table_equal(table, expected)

@pytest.mark.parametrize('parallel', [True, False])
def test_no_header(parallel, read_basic, read_no_header):
    if False:
        return 10
    '\n    The header should not be read when header_start=None. Unless names is\n    passed, the column names should be auto-generated.\n    '
    with pytest.raises(ValueError):
        read_basic('A B C\n1 2 3\n4 5 6', header_start=None, data_start=0, parallel=parallel)
    t2 = read_no_header('A B C\n1 2 3\n4 5 6', parallel=parallel)
    expected = Table([['A', '1', '4'], ['B', '2', '5'], ['C', '3', '6']], names=('col1', 'col2', 'col3'))
    assert_table_equal(t2, expected)

@pytest.mark.parametrize('parallel', [True, False])
def test_no_header_supplied_names(parallel, read_basic, read_no_header):
    if False:
        print('Hello World!')
    '\n    If header_start=None and names is passed as a parameter, header\n    data should not be read and names should be used instead.\n    '
    table = read_no_header('A B C\n1 2 3\n4 5 6', names=('X', 'Y', 'Z'), parallel=parallel)
    expected = Table([['A', '1', '4'], ['B', '2', '5'], ['C', '3', '6']], names=('X', 'Y', 'Z'))
    assert_table_equal(table, expected)

@pytest.mark.parametrize('parallel', [True, False])
def test_comment(parallel, read_basic):
    if False:
        return 10
    '\n    Make sure that line comments are ignored by the C reader.\n    '
    table = read_basic('# comment\nA B C\n # another comment\n1 2 3\n4 5 6', parallel=parallel)
    expected = Table([[1, 4], [2, 5], [3, 6]], names=('A', 'B', 'C'))
    assert_table_equal(table, expected)

@pytest.mark.parametrize('parallel', [True, False])
def test_empty_lines(parallel, read_basic):
    if False:
        print('Hello World!')
    '\n    Make sure that empty lines are ignored by the C reader.\n    '
    table = read_basic('\n\nA B C\n1 2 3\n\n\n4 5 6\n\n\n\n', parallel=parallel)
    expected = Table([[1, 4], [2, 5], [3, 6]], names=('A', 'B', 'C'))
    assert_table_equal(table, expected)

@pytest.mark.parametrize('parallel', [True, False])
def test_lstrip_whitespace(parallel, read_basic):
    if False:
        print('Hello World!')
    '\n    Test to make sure the reader ignores whitespace at the beginning of fields.\n    '
    text = '\n     1,  2,   \t3\n A,\t\t B,  C\n  a, b,   c\n  \n'
    table = read_basic(text, delimiter=',', parallel=parallel)
    expected = Table([['A', 'a'], ['B', 'b'], ['C', 'c']], names=('1', '2', '3'))
    assert_table_equal(table, expected)

@pytest.mark.parametrize('parallel', [True, False])
def test_rstrip_whitespace(parallel, read_basic):
    if False:
        print('Hello World!')
    '\n    Test to make sure the reader ignores whitespace at the end of fields.\n    '
    text = ' 1 ,2 \t,3  \nA\t,B ,C\t \t \n  \ta ,b , c \n'
    table = read_basic(text, delimiter=',', parallel=parallel)
    expected = Table([['A', 'a'], ['B', 'b'], ['C', 'c']], names=('1', '2', '3'))
    assert_table_equal(table, expected)

@pytest.mark.parametrize('parallel', [True, False])
def test_conversion(parallel, read_basic):
    if False:
        while True:
            i = 10
    '\n    The reader should try to convert each column to ints. If this fails, the\n    reader should try to convert to floats. Failing this, i.e. on parsing\n    non-numeric input including isolated positive/negative signs, it should\n    fall back to strings.\n    '
    text = '\nA B C D E F G H\n1 a 3 4 5 6 7 8\n2. 1 9 -.1e1 10.0 8.7 6 -5.3e4\n4 2 -12 .4 +.e1 - + six\n'
    table = read_basic(text, parallel=parallel)
    assert_equal(table['A'].dtype.kind, 'f')
    assert table['B'].dtype.kind in ('S', 'U')
    assert_equal(table['C'].dtype.kind, 'i')
    assert_equal(table['D'].dtype.kind, 'f')
    assert table['E'].dtype.kind in ('S', 'U')
    assert table['F'].dtype.kind in ('S', 'U')
    assert table['G'].dtype.kind in ('S', 'U')
    assert table['H'].dtype.kind in ('S', 'U')

@pytest.mark.parametrize('parallel', [True, False])
def test_delimiter(parallel, read_basic):
    if False:
        print('Hello World!')
    '\n    Make sure that different delimiters work as expected.\n    '
    text = dedent('\n    COL1 COL2 COL3\n    1 A -1\n    2 B -2\n    ')
    expected = Table([[1, 2], ['A', 'B'], [-1, -2]], names=('COL1', 'COL2', 'COL3'))
    for sep in ' ,\t#;':
        table = read_basic(text.replace(' ', sep), delimiter=sep, parallel=parallel)
        assert_table_equal(table, expected)

@pytest.mark.parametrize('parallel', [True, False])
def test_include_names(parallel, read_basic):
    if False:
        while True:
            i = 10
    '\n    If include_names is not None, the parser should read only those columns in include_names.\n    '
    table = read_basic('A B C D\n1 2 3 4\n5 6 7 8', include_names=['A', 'D'], parallel=parallel)
    expected = Table([[1, 5], [4, 8]], names=('A', 'D'))
    assert_table_equal(table, expected)

@pytest.mark.parametrize('parallel', [True, False])
def test_exclude_names(parallel, read_basic):
    if False:
        i = 10
        return i + 15
    '\n    If exclude_names is not None, the parser should exclude the columns in exclude_names.\n    '
    table = read_basic('A B C D\n1 2 3 4\n5 6 7 8', exclude_names=['A', 'D'], parallel=parallel)
    expected = Table([[2, 6], [3, 7]], names=('B', 'C'))
    assert_table_equal(table, expected)

@pytest.mark.parametrize('parallel', [True, False])
def test_include_exclude_names(parallel, read_basic):
    if False:
        i = 10
        return i + 15
    '\n    Make sure that include_names is applied before exclude_names if both are specified.\n    '
    text = dedent('\n    A B C D E F G H\n    1 2 3 4 5 6 7 8\n    9 10 11 12 13 14 15 16\n    ')
    table = read_basic(text, include_names=['A', 'B', 'D', 'F', 'H'], exclude_names=['B', 'F'], parallel=parallel)
    expected = Table([[1, 9], [4, 12], [8, 16]], names=('A', 'D', 'H'))
    assert_table_equal(table, expected)

def test_doubled_quotes(read_csv):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test #8283 (fix for #8281), parsing doubled-quotes "ab""cd" in a quoted\n    field was incorrect.\n\n    '
    tbl = '\n'.join(['a,b', '"d""","d""q"', '"""q",""""'])
    expected = Table([['d"', '"q'], ['d"q', '"']], names=('a', 'b'))
    dat = read_csv(tbl)
    assert_table_equal(dat, expected)
    for fast_reader in (True, False):
        dat = ascii.read(tbl, fast_reader=fast_reader)
        assert_table_equal(dat, expected)

@pytest.mark.filterwarnings('ignore:OverflowError converting to IntType in column TIMESTAMP')
def test_doubled_quotes_segv():
    if False:
        print('Hello World!')
    '\n    Test the exact example from #8281 which resulted in SEGV prior to #8283\n    (in contrast to the tests above that just gave the wrong answer).\n    Attempts to produce a more minimal example were unsuccessful, so the whole\n    thing is included.\n    '
    tbl = dedent('\n    "ID","TIMESTAMP","addendum_id","bib_reference","bib_reference_url","client_application","client_category","client_sort_key","color","coordsys","creator","creator_did","data_pixel_bitpix","dataproduct_subtype","dataproduct_type","em_max","em_min","format","hips_builder","hips_copyright","hips_creation_date","hips_creation_date_1","hips_creator","hips_data_range","hips_estsize","hips_frame","hips_glu_tag","hips_hierarchy","hips_initial_dec","hips_initial_fov","hips_initial_ra","hips_lon_asc","hips_master_url","hips_order","hips_order_1","hips_order_4","hips_order_min","hips_overlay","hips_pixel_bitpix","hips_pixel_cut","hips_pixel_scale","hips_progenitor_url","hips_publisher","hips_release_date","hips_release_date_1","hips_rgb_blue","hips_rgb_green","hips_rgb_red","hips_sampling","hips_service_url","hips_service_url_1","hips_service_url_2","hips_service_url_3","hips_service_url_4","hips_service_url_5","hips_service_url_6","hips_service_url_7","hips_service_url_8","hips_skyval","hips_skyval_method","hips_skyval_value","hips_status","hips_status_1","hips_status_2","hips_status_3","hips_status_4","hips_status_5","hips_status_6","hips_status_7","hips_status_8","hips_tile_format","hips_tile_format_1","hips_tile_format_4","hips_tile_width","hips_version","hipsgen_date","hipsgen_date_1","hipsgen_date_10","hipsgen_date_11","hipsgen_date_12","hipsgen_date_2","hipsgen_date_3","hipsgen_date_4","hipsgen_date_5","hipsgen_date_6","hipsgen_date_7","hipsgen_date_8","hipsgen_date_9","hipsgen_params","hipsgen_params_1","hipsgen_params_10","hipsgen_params_11","hipsgen_params_12","hipsgen_params_2","hipsgen_params_3","hipsgen_params_4","hipsgen_params_5","hipsgen_params_6","hipsgen_params_7","hipsgen_params_8","hipsgen_params_9","label","maxOrder","moc_access_url","moc_order","moc_release_date","moc_sky_fraction","obs_ack","obs_collection","obs_copyrigh_url","obs_copyright","obs_copyright_1","obs_copyright_url","obs_copyright_url_1","obs_description","obs_description_url","obs_descrition_url","obs_id","obs_initial_dec","obs_initial_fov","obs_initial_ra","obs_provenance","obs_regime","obs_title","ohips_frame","pixelCut","pixelRange","prov_did","prov_progenitor","prov_progenitor_url","publisher_did","publisher_id","s_pixel_scale","t_max","t_min"\n    "CDS/P/2MASS/H","1524123841000","","2006AJ....131.1163S","http://cdsbib.unistra.fr/cgi-bin/cdsbib?2006AJ....131.1163S","AladinDesktop","Image/Infrared/2MASS","04-001-03","","","","ivo://CDS/P/2MASS/H","","","image","1.798E-6","1.525E-6","","Aladin/HipsGen v9.017","CNRS/Unistra","2013-05-06T20:36Z","","CDS (A.Oberto)","","","equatorial","","mean","","","","","","9","","","","","","0 60","2.236E-4","","","2016-04-22T13:48Z","","","","","","http://alasky.unistra.fr/2MASS/H","https://irsa.ipac.caltech.edu/data/hips/CDS/2MASS/H","http://alaskybis.unistra.fr/2MASS/H","https://alaskybis.unistra.fr/2MASS/H","","","","","","","","","public master clonableOnce","public mirror unclonable","public mirror clonableOnce","public mirror clonableOnce","","","","","","jpeg fits","","","512","1.31","","","","","","","","","","","","","","","","","","","","","","","","","","","","","http://alasky.unistra.fr/2MASS/H/Moc.fits","9","","1","University of Massachusetts & IPAC/Caltech","The Two Micron All Sky Survey - H band (2MASS H)","","University of Massachusetts & IPAC/Caltech","","http://www.ipac.caltech.edu/2mass/","","2MASS has uniformly scanned the entire sky in three near-infrared bands to detect and characterize point sources brighter than about 1 mJy in each band, with signal-to-noise ratio (SNR) greater than 10, using a pixel size of 2.0"". This has achieved an 80,000-fold improvement in sensitivity relative to earlier surveys. 2MASS used two highly-automated 1.3-m telescopes, one at Mt. Hopkins, AZ, and one at CTIO, Chile. Each telescope was equipped with a three-channel camera, each channel consisting of a 256x256 array of HgCdTe detectors, capable of observing the sky simultaneously at J (1.25 microns), H (1.65 microns), and Ks (2.17 microns). The University of Massachusetts (UMass) was responsible for the overall management of the project, and for developing the infrared cameras and on-site computing systems at both facilities. The Infrared Processing and Analysis Center (IPAC) is responsible for all data processing through the Production Pipeline, and construction and distribution of the data products. Funding is provided primarily by NASA and the NSF","","","","+0","0.11451621372724685","0","","Infrared","2MASS H (1.66um)","","","","","IPAC/NASA","","","","","51941","50600"\n    ')
    ascii.read(tbl, format='csv', fast_reader=True, guess=False)

@pytest.mark.parametrize('parallel', [True, False])
def test_quoted_fields(parallel, read_basic):
    if False:
        while True:
            i = 10
    '\n    The character quotechar (default \'"\') should denote the start of a field which can\n    contain the field delimiter and newlines.\n    '
    if parallel:
        pytest.xfail('Multiprocessing can fail with quoted fields')
    text = dedent('\n    "A B" C D\n    1.5 2.1 -37.1\n    a b "   c\n    d"\n    ')
    table = read_basic(text, parallel=parallel)
    expected = Table([['1.5', 'a'], ['2.1', 'b'], ['-37.1', 'c\nd']], names=('A B', 'C', 'D'))
    assert_table_equal(table, expected)
    table = read_basic(text.replace('"', "'"), quotechar="'", parallel=parallel)
    assert_table_equal(table, expected)

@pytest.mark.parametrize('key,val', [('delimiter', ',,'), ('comment', '##'), ('data_start', None), ('data_start', -1), ('quotechar', '##'), ('header_start', -1), ('converters', {i + 1: ascii.convert_numpy(np.uint) for i in range(3)}), ('inputter_cls', ascii.ContinuationLinesInputter), ('header_splitter_cls', ascii.DefaultSplitter), ('data_splitter_cls', ascii.DefaultSplitter)])
def test_invalid_parameters(key, val):
    if False:
        while True:
            i = 10
    "\n    Make sure the C reader raises an error if passed parameters it can't handle.\n    "
    with pytest.raises(ParameterError):
        FastBasic(**{key: val}).read('1 2 3\n4 5 6')
    with pytest.raises(ParameterError):
        ascii.read('1 2 3\n4 5 6', format='fast_basic', guess=False, **{key: val})

def test_invalid_parameters_other():
    if False:
        i = 10
        return i + 15
    with pytest.raises(TypeError):
        FastBasic(foo=7).read('1 2 3\n4 5 6')
    with pytest.raises(FastOptionsError):
        ascii.read('1 2 3\n4 5 6', format='basic', fast_reader={'foo': 7})
    with pytest.raises(ParameterError):
        FastBasic(outputter_cls=ascii.TableOutputter).read('1 2 3\n4 5 6')

def test_too_many_cols1():
    if False:
        print('Hello World!')
    '\n    If a row contains too many columns, the C reader should raise an error.\n    '
    text = dedent('\n    A B C\n    1 2 3\n    4 5 6\n    7 8 9 10\n    11 12 13\n    ')
    with pytest.raises(InconsistentTableError) as e:
        FastBasic().read(text)
    assert 'Number of header columns (3) inconsistent with data columns in data line 2' in str(e.value)

def test_too_many_cols2():
    if False:
        while True:
            i = 10
    text = 'aaa,bbb\n1,2,\n3,4,\n'
    with pytest.raises(InconsistentTableError) as e:
        FastCsv().read(text)
    assert 'Number of header columns (2) inconsistent with data columns in data line 0' in str(e.value)

def test_too_many_cols3():
    if False:
        for i in range(10):
            print('nop')
    text = 'aaa,bbb\n1,2,,\n3,4,\n'
    with pytest.raises(InconsistentTableError) as e:
        FastCsv().read(text)
    assert 'Number of header columns (2) inconsistent with data columns in data line 0' in str(e.value)

def test_too_many_cols4():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(InconsistentTableError) as e:
        ascii.read(get_pkg_data_filename('data/conf_py.txt'), fast_reader=True, guess=True)
    assert 'Unable to guess table format with the guesses listed below' in str(e.value)

@pytest.mark.parametrize('parallel', [True, False])
def test_not_enough_cols(parallel, read_csv):
    if False:
        print('Hello World!')
    '\n    If a row does not have enough columns, the FastCsv reader should add empty\n    fields while the FastBasic reader should raise an error.\n    '
    text = '\nA,B,C\n1,2,3\n4,5\n6,7,8\n'
    table = read_csv(text, parallel=parallel)
    assert table['B'][1] is not ma.masked
    assert table['C'][1] is ma.masked
    with pytest.raises(InconsistentTableError):
        table = FastBasic(delimiter=',').read(text)

@pytest.mark.parametrize('parallel', [True, False])
def test_data_end(parallel, read_basic, read_rdb):
    if False:
        return 10
    '\n    The parameter data_end should specify where data reading ends.\n    '
    text = '\nA B C\n1 2 3\n4 5 6\n7 8 9\n10 11 12\n'
    table = read_basic(text, data_end=3, parallel=parallel)
    expected = Table([[1, 4], [2, 5], [3, 6]], names=('A', 'B', 'C'))
    assert_table_equal(table, expected)
    table = read_basic(text, data_end=-2, parallel=parallel)
    assert_table_equal(table, expected)
    text = '\nA\tB\tC\nN\tN\tS\n1\t2\ta\n3\t4\tb\n5\t6\tc\n'
    table = read_rdb(text, data_end=-1, parallel=parallel)
    expected = Table([[1, 3], [2, 4], ['a', 'b']], names=('A', 'B', 'C'))
    assert_table_equal(table, expected)
    table = read_rdb(text, data_end=3, parallel=parallel)
    expected = Table([[1], [2], ['a']], names=('A', 'B', 'C'))
    assert_table_equal(table, expected)
    table = read_rdb(text, data_end=1, parallel=parallel)
    expected = Table([[], [], []], names=('A', 'B', 'C'))
    assert_table_equal(table, expected)

@pytest.mark.parametrize('parallel', [True, False])
def test_inf_nan(parallel, read_basic):
    if False:
        print('Hello World!')
    '\n    Test that inf and nan-like values are correctly parsed on all platforms.\n\n    Regression test for https://github.com/astropy/astropy/pull/3525\n    '
    text = dedent('        A\n        nan\n        +nan\n        -nan\n        inf\n        infinity\n        +inf\n        +infinity\n        -inf\n        -infinity\n    ')
    expected = Table({'A': [np.nan, np.nan, np.nan, np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf]})
    table = read_basic(text, parallel=parallel)
    assert table['A'].dtype.kind == 'f'
    assert_table_equal(table, expected)

@pytest.mark.parametrize('parallel', [True, False])
def test_fill_values(parallel, read_basic):
    if False:
        while True:
            i = 10
    "\n    Make sure that the parameter fill_values works as intended. If fill_values\n    is not specified, the default behavior should be to convert '' to 0.\n    "
    text = '\nA, B, C\n, 2, nan\na, -999, -3.4\nnan, 5, -9999\n8, nan, 7.6e12\n'
    table = read_basic(text, delimiter=',', parallel=parallel)
    assert isinstance(table['A'], MaskedColumn)
    assert table['A'][0] is ma.masked
    assert_equal(table['A'].data.data[0], '0')
    assert table['A'][1] is not ma.masked
    table = read_basic(text, delimiter=',', fill_values=('-999', '0'), parallel=parallel)
    assert isinstance(table['B'], MaskedColumn)
    assert table['A'][0] is not ma.masked
    assert table['C'][2] is not ma.masked
    assert table['B'][1] is ma.masked
    assert_equal(table['B'].data.data[1], 0.0)
    assert table['B'][0] is not ma.masked
    table = read_basic(text, delimiter=',', fill_values=[], parallel=parallel)
    for name in 'ABC':
        assert not isinstance(table[name], MaskedColumn)
    table = read_basic(text, delimiter=',', fill_values=[('', '0', 'A'), ('nan', '999', 'A', 'C')], parallel=parallel)
    assert np.isnan(table['B'][3])
    assert table['B'][3] is not ma.masked
    assert table['A'][0] is ma.masked
    assert table['A'][2] is ma.masked
    assert_equal(table['A'].data.data[0], '0')
    assert_equal(table['A'].data.data[2], '999')
    assert table['C'][0] is ma.masked
    assert_almost_equal(table['C'].data.data[0], 999.0)
    assert_almost_equal(table['C'][1], -3.4)

@pytest.mark.parametrize('parallel', [True, False])
def test_fill_include_exclude_names(parallel, read_csv):
    if False:
        while True:
            i = 10
    '\n    fill_include_names and fill_exclude_names should filter missing/empty value handling\n    in the same way that include_names and exclude_names filter output columns.\n    '
    text = '\nA, B, C\n, 1, 2\n3, , 4\n5, 5,\n'
    table = read_csv(text, fill_include_names=['A', 'B'], parallel=parallel)
    assert table['A'][0] is ma.masked
    assert table['B'][1] is ma.masked
    assert table['C'][2] is not ma.masked
    table = read_csv(text, fill_exclude_names=['A', 'B'], parallel=parallel)
    assert table['C'][2] is ma.masked
    assert table['A'][0] is not ma.masked
    assert table['B'][1] is not ma.masked
    table = read_csv(text, fill_include_names=['A', 'B'], fill_exclude_names=['B'], parallel=parallel)
    assert table['A'][0] is ma.masked
    assert table['B'][1] is not ma.masked
    assert table['C'][2] is not ma.masked

@pytest.mark.parametrize('parallel', [True, False])
def test_many_rows(parallel, read_basic):
    if False:
        while True:
            i = 10
    '\n    Make sure memory reallocation works okay when the number of rows\n    is large (so that each column string is longer than INITIAL_COL_SIZE).\n    '
    text = 'A B C\n'
    for i in range(500):
        text += ' '.join([str(i) for i in range(3)])
        text += '\n'
    table = read_basic(text, parallel=parallel)
    expected = Table([[0] * 500, [1] * 500, [2] * 500], names=('A', 'B', 'C'))
    assert_table_equal(table, expected)

@pytest.mark.parametrize('parallel', [True, False])
def test_many_columns(parallel, read_basic):
    if False:
        i = 10
        return i + 15
    '\n    Make sure memory reallocation works okay when the number of columns\n    is large (so that each header string is longer than INITIAL_HEADER_SIZE).\n    '
    text = ' '.join([str(i) for i in range(500)])
    text += '\n' + text + '\n' + text
    table = read_basic(text, parallel=parallel)
    expected = Table([[i, i] for i in range(500)], names=[str(i) for i in range(500)])
    assert_table_equal(table, expected)

def test_fast_reader():
    if False:
        return 10
    '\n    Make sure that ascii.read() works as expected by default and with\n    fast_reader specified.\n    '
    text = 'a b c\n1 2 3\n4 5 6'
    with pytest.raises(ParameterError):
        ascii.read(text, format='fast_basic', guess=False, comment='##')
    try:
        ascii.read(text, format='basic', guess=False, fast_reader={'parallel': True, 'use_fast_converter': True})
    except NotImplementedError:
        if os.name == 'nt':
            ascii.read(text, format='basic', guess=False, fast_reader={'parallel': False, 'use_fast_converter': True})
        else:
            raise
    with pytest.raises(FastOptionsError):
        ascii.read(text, format='fast_basic', guess=False, fast_reader={'foo': True})
    ascii.read(text, format='basic', guess=False, comment='##', fast_reader=False)
    ascii.read(text, format='basic', guess=False, comment='##')

@pytest.mark.parametrize('parallel', [True, False])
def test_read_tab(parallel, read_tab):
    if False:
        while True:
            i = 10
    '\n    The fast reader for tab-separated values should not strip whitespace, unlike\n    the basic reader.\n    '
    if parallel:
        pytest.xfail('Multiprocessing can fail with quoted fields')
    text = '1\t2\t3\n  a\t b \t\n c\t" d\n e"\t  '
    table = read_tab(text, parallel=parallel)
    assert_equal(table['1'][0], '  a')
    assert_equal(table['2'][0], ' b ')
    assert table['3'][0] is ma.masked
    assert_equal(table['2'][1], ' d\n e')
    assert_equal(table['3'][1], '  ')

@pytest.mark.parametrize('parallel', [True, False])
def test_default_data_start(parallel, read_basic):
    if False:
        i = 10
        return i + 15
    '\n    If data_start is not explicitly passed to read(), data processing should\n    beginning right after the header.\n    '
    text = 'ignore this line\na b c\n1 2 3\n4 5 6'
    table = read_basic(text, header_start=1, parallel=parallel)
    expected = Table([[1, 4], [2, 5], [3, 6]], names=('a', 'b', 'c'))
    assert_table_equal(table, expected)

@pytest.mark.parametrize('parallel', [True, False])
def test_commented_header(parallel, read_commented_header):
    if False:
        return 10
    '\n    The FastCommentedHeader reader should mimic the behavior of the\n    CommentedHeader by overriding the default header behavior of FastBasic.\n    '
    text = '\n # A B C\n 1 2 3\n 4 5 6\n'
    t1 = read_commented_header(text, parallel=parallel)
    expected = Table([[1, 4], [2, 5], [3, 6]], names=('A', 'B', 'C'))
    assert_table_equal(t1, expected)
    text = '# first commented line\n # second commented line\n\n' + text
    t2 = read_commented_header(text, header_start=2, data_start=0, parallel=parallel)
    assert_table_equal(t2, expected)
    t3 = read_commented_header(text, header_start=-1, data_start=0, parallel=parallel)
    assert_table_equal(t3, expected)
    text += '7 8 9'
    t4 = read_commented_header(text, header_start=2, data_start=2, parallel=parallel)
    expected = Table([[7], [8], [9]], names=('A', 'B', 'C'))
    assert_table_equal(t4, expected)
    with pytest.raises(ParameterError):
        read_commented_header(text, header_start=-1, data_start=-1, parallel=parallel)

@pytest.mark.parametrize('parallel', [True, False])
def test_rdb(parallel, read_rdb):
    if False:
        i = 10
        return i + 15
    '\n    Make sure the FastRdb reader works as expected.\n    '
    text = '\n\nA\tB\tC\n1n\tS\t4N\n1\t 9\t4.3\n'
    table = read_rdb(text, parallel=parallel)
    expected = Table([[1], [' 9'], [4.3]], names=('A', 'B', 'C'))
    assert_table_equal(table, expected)
    assert_equal(table['A'].dtype.kind, 'i')
    assert table['B'].dtype.kind in ('S', 'U')
    assert_equal(table['C'].dtype.kind, 'f')
    with pytest.raises(ValueError) as e:
        text = 'A\tB\tC\nN\tS\tN\n4\tb\ta'
        read_rdb(text, parallel=parallel)
    assert 'Column C failed to convert' in str(e.value)
    with pytest.raises(ValueError) as e:
        text = 'A\tB\tC\nN\tN\n1\t2\t3'
        read_rdb(text, parallel=parallel)
    assert 'mismatch between number of column names and column types' in str(e.value)
    with pytest.raises(ValueError) as e:
        text = 'A\tB\tC\nN\tN\t5\n1\t2\t3'
        read_rdb(text, parallel=parallel)
    assert 'type definitions do not all match [num](N|S)' in str(e.value)

@pytest.mark.parametrize('parallel', [True, False])
def test_data_start(parallel, read_basic):
    if False:
        return 10
    '\n    Make sure that data parsing begins at data_start (ignoring empty and\n    commented lines but not taking quoted values into account).\n    '
    if parallel:
        pytest.xfail('Multiprocessing can fail with quoted fields')
    text = '\nA B C\n1 2 3\n4 5 6\n\n7 8 "9\n1"\n# comment\n10 11 12\n'
    table = read_basic(text, data_start=2, parallel=parallel)
    expected = Table([[4, 7, 10], [5, 8, 11], ['6', '9\n1', '12']], names=('A', 'B', 'C'))
    assert_table_equal(table, expected)
    table = read_basic(text, data_start=3, parallel=parallel)
    expected = Table([[7, 10], [8, 11], ['9\n1', '12']], names=('A', 'B', 'C'))
    assert_table_equal(table, expected)
    with pytest.raises(InconsistentTableError) as e:
        read_basic(text, data_start=4, parallel=parallel)
    assert 'header columns (3) inconsistent with data columns in data line 0' in str(e.value)
    table = read_basic(text, data_start=5, parallel=parallel)
    expected = Table([[10], [11], [12]], names=('A', 'B', 'C'))
    assert_table_equal(table, expected)
    text = '\nA B C\n1 2 3\n4 5 6\n\n7 8 9\n# comment\n10 11 12\n'
    table = read_basic(text, data_start=2, parallel=parallel)
    expected = Table([[4, 7, 10], [5, 8, 11], [6, 9, 12]], names=('A', 'B', 'C'))
    assert_table_equal(table, expected)

@pytest.mark.parametrize('parallel', [True, False])
def test_quoted_empty_values(parallel, read_basic):
    if False:
        for i in range(10):
            print('nop')
    '\n    Quoted empty values spanning multiple lines should be treated correctly.\n    '
    if parallel:
        pytest.xfail('Multiprocessing can fail with quoted fields')
    text = 'a b c\n1 2 " \n "'
    table = read_basic(text, parallel=parallel)
    assert table['c'][0] == '\n'

@pytest.mark.parametrize('parallel', [True, False])
def test_csv_comment_default(parallel, read_csv):
    if False:
        print('Hello World!')
    '\n    Unless the comment parameter is specified, the CSV reader should\n    not treat any lines as comments.\n    '
    text = 'a,b,c\n#1,2,3\n4,5,6'
    table = read_csv(text, parallel=parallel)
    expected = Table([['#1', '4'], [2, 5], [3, 6]], names=('a', 'b', 'c'))
    assert_table_equal(table, expected)

@pytest.mark.parametrize('parallel', [True, False])
def test_whitespace_before_comment(parallel, read_tab):
    if False:
        i = 10
        return i + 15
    "\n    Readers that don't strip whitespace from data (Tab, RDB)\n    should still treat lines with leading whitespace and then\n    the comment char as comment lines.\n    "
    text = 'a\tb\tc\n # comment line\n1\t2\t3'
    table = read_tab(text, parallel=parallel)
    expected = Table([[1], [2], [3]], names=('a', 'b', 'c'))
    assert_table_equal(table, expected)

@pytest.mark.parametrize('parallel', [True, False])
def test_strip_line_trailing_whitespace(parallel, read_basic):
    if False:
        while True:
            i = 10
    '\n    Readers that strip whitespace from lines should ignore\n    trailing whitespace after the last data value of each\n    row.\n    '
    text = 'a b c\n1 2 \n3 4 5'
    with pytest.raises(InconsistentTableError) as e:
        ascii.read(StringIO(text), format='fast_basic', guess=False)
    assert 'header columns (3) inconsistent with data columns in data line 0' in str(e.value)
    text = 'a b c\n 1 2 3   \t \n 4 5 6 '
    table = read_basic(text, parallel=parallel)
    expected = Table([[1, 4], [2, 5], [3, 6]], names=('a', 'b', 'c'))
    assert_table_equal(table, expected)

@pytest.mark.parametrize('parallel', [True, False])
def test_no_data(parallel, read_basic):
    if False:
        return 10
    '\n    As long as column names are supplied, the C reader\n    should return an empty table in the absence of data.\n    '
    table = read_basic('a b c', parallel=parallel)
    expected = Table([[], [], []], names=('a', 'b', 'c'))
    assert_table_equal(table, expected)
    table = read_basic('a b c\n1 2 3', data_start=2, parallel=parallel)
    assert_table_equal(table, expected)

@pytest.mark.parametrize('parallel', [True, False])
def test_line_endings(parallel, read_basic, read_commented_header, read_rdb):
    if False:
        i = 10
        return i + 15
    '\n    Make sure the fast reader accepts CR and CR+LF\n    as newlines.\n    '
    text = 'a b c\n1 2 3\n4 5 6\n7 8 9\n'
    expected = Table([[1, 4, 7], [2, 5, 8], [3, 6, 9]], names=('a', 'b', 'c'))
    for newline in ('\r\n', '\r'):
        table = read_basic(text.replace('\n', newline), parallel=parallel)
        assert_table_equal(table, expected)
    text = '#' + text
    for newline in ('\r\n', '\r'):
        table = read_commented_header(text.replace('\n', newline), parallel=parallel)
        assert_table_equal(table, expected)
    expected = Table([MaskedColumn([1, 4, 7]), [2, 5, 8], MaskedColumn([3, 6, 9])], names=('a', 'b', 'c'))
    expected['a'][0] = np.ma.masked
    expected['c'][0] = np.ma.masked
    text = 'a\tb\tc\nN\tN\tN\n\t2\t\n4\t5\t6\n7\t8\t9\n'
    for newline in ('\r\n', '\r'):
        table = read_rdb(text.replace('\n', newline), parallel=parallel)
        assert_table_equal(table, expected)
        assert np.all(table == expected)

@pytest.mark.parametrize('parallel', [True, False])
def test_store_comments(parallel, read_basic):
    if False:
        i = 10
        return i + 15
    '\n    Make sure that the output Table produced by the fast\n    reader stores any comment lines in its meta attribute.\n    '
    text = '\n# header comment\na b c\n# comment 2\n# comment 3\n1 2 3\n4 5 6\n'
    table = read_basic(text, parallel=parallel, check_meta=True)
    assert_equal(table.meta['comments'], ['header comment', 'comment 2', 'comment 3'])

@pytest.mark.parametrize('parallel', [True, False])
def test_empty_quotes(parallel, read_basic):
    if False:
        while True:
            i = 10
    "\n    Make sure the C reader doesn't segfault when the\n    input data contains empty quotes. [#3407]\n    "
    table = read_basic('a b\n1 ""\n2 ""', parallel=parallel)
    expected = Table([[1, 2], [0, 0]], names=('a', 'b'))
    assert_table_equal(table, expected)

@pytest.mark.parametrize('parallel', [True, False])
def test_fast_tab_with_names(parallel, read_tab):
    if False:
        for i in range(10):
            print('nop')
    "\n    Make sure the C reader doesn't segfault when the header for the\n    first column is missing [#3545]\n    "
    content = '#\n\tdecDeg\tRate_pn_offAxis\tRate_mos2_offAxis\tObsID\tSourceID\tRADeg\tversion\tCounts_pn\tRate_pn\trun\tRate_mos1\tRate_mos2\tInserted_pn\tInserted_mos2\tbeta\tRate_mos1_offAxis\trcArcsec\tname\tInserted\tCounts_mos1\tInserted_mos1\tCounts_mos2\ty\tx\tCounts\toffAxis\tRot\n-3.007559\t0.0000\t0.0010\t0013140201\t0\t213.462574\t0\t2\t0.0002\t0\t0.0001\t0.0001\t0\t1\t0.66\t0.0217\t3.0\tfakeXMMXCS J1413.8-0300\t3\t1\t2\t1\t398.000\t127.000\t5\t13.9\t72.3\t'
    head = [f'A{i}' for i in range(28)]
    read_tab(content, data_start=1, parallel=parallel, names=head)

@pytest.mark.hugemem
def test_read_big_table(tmp_path):
    if False:
        while True:
            i = 10
    'Test reading of a huge file.\n\n    This test generates a huge CSV file (~2.3Gb) before reading it (see\n    https://github.com/astropy/astropy/pull/5319). The test is run only if the\n    ``--run-hugemem`` cli option is given. Note that running the test requires\n    quite a lot of memory (~18Gb when reading the file) !!\n\n    '
    NB_ROWS = 250000
    NB_COLS = 500
    filename = tmp_path / 'big_table.csv'
    print(f'Creating a {NB_ROWS} rows table ({NB_COLS} columns).')
    data = np.random.random(NB_ROWS)
    t = Table(data=[data] * NB_COLS, names=[str(i) for i in range(NB_COLS)])
    data = None
    print(f'Saving the table to {filename}')
    t.write(filename, format='ascii.csv', overwrite=True)
    t = None
    print('Counting the number of lines in the csv, it should be {NB_ROWS} + 1 (header).')
    with open(filename) as f:
        assert sum((1 for line in f)) == NB_ROWS + 1
    print('Reading the file with astropy.')
    t = Table.read(filename, format='ascii.csv', fast_reader=True)
    assert len(t) == NB_ROWS

@pytest.mark.hugemem
def test_read_big_table2(tmp_path):
    if False:
        i = 10
        return i + 15
    'Test reading of a file with a huge column.'
    NB_ROWS = 2 ** 32 // 2 // 10 + 5
    filename = tmp_path / 'big_table.csv'
    print(f'Creating a {NB_ROWS} rows table.')
    data = np.full(NB_ROWS, int(1000000000.0), dtype=np.int32)
    t = Table(data=[data], names=['a'], copy=False)
    print(f'Saving the table to {filename}')
    t.write(filename, format='ascii.csv', overwrite=True)
    t = None
    print('Counting the number of lines in the csv, it should be {NB_ROWS} + 1 (header).')
    with open(filename) as f:
        assert sum((1 for line in f)) == NB_ROWS + 1
    print('Reading the file with astropy.')
    t = Table.read(filename, format='ascii.csv', fast_reader=True)
    assert len(t) == NB_ROWS

@pytest.mark.parametrize('guess', [True, False])
@pytest.mark.parametrize('fast_reader', [False, {'use_fast_converter': False}, {'use_fast_converter': True}])
@pytest.mark.parametrize('parallel', [False, True])
def test_data_out_of_range(parallel, fast_reader, guess):
    if False:
        i = 10
        return i + 15
    '\n    Numbers with exponents beyond float64 range (|~4.94e-324 to 1.7977e+308|)\n    shall be returned as 0 and +-inf respectively by the C parser, just like\n    the Python parser.\n    Test fast converter only to nominal accuracy.\n    '
    rtol = 1e-30
    if fast_reader:
        fast_reader['parallel'] = parallel
        if fast_reader.get('use_fast_converter'):
            rtol = 1e-15
        elif sys.maxsize < 2 ** 32:
            pytest.xfail('C parser cannot handle float64 on 32bit systems')
    if parallel:
        if not fast_reader:
            pytest.skip('Multiprocessing only available in fast reader')
        elif CI:
            pytest.xfail('Multiprocessing can sometimes fail on CI')
    test_for_warnings = fast_reader and (not parallel)
    if not parallel and (not fast_reader):
        ctx = nullcontext()
    else:
        ctx = pytest.warns()
    fields = ['10.1E+199', '3.14e+313', '2048e+306', '0.6E-325', '-2.e345']
    values = np.array([1.01e+200, np.inf, np.inf, 0.0, -np.inf])
    with ctx as w:
        t = ascii.read(StringIO(' '.join(fields)), format='no_header', guess=guess, fast_reader=fast_reader)
    if test_for_warnings:
        assert len(w) == 4
        for i in range(len(w)):
            assert f'OverflowError converting to FloatType in column col{i + 2}' in str(w[i].message)
    read_values = np.array([col[0] for col in t.itercols()])
    assert_almost_equal(read_values, values, rtol=rtol, atol=0.0)
    fields = ['.0101E202', '0.000000314E+314', '1777E+305', '-1799E+305', '0.2e-323', '5200e-327', ' 0.0000000000000000000001024E+330']
    values = np.array([1.01e+200, 3.14e+307, 1.777e+308, -np.inf, 0.0, 5e-324, 1.024e+308])
    with ctx as w:
        t = ascii.read(StringIO(' '.join(fields)), format='no_header', guess=guess, fast_reader=fast_reader)
    if test_for_warnings:
        if sys.platform == 'win32' and (not fast_reader.get('use_fast_converter')):
            assert len(w) == 2
        else:
            assert len(w) == 3
        for i in range(len(w)):
            assert f'OverflowError converting to FloatType in column col{i + 4}' in str(w[i].message)
    read_values = np.array([col[0] for col in t.itercols()])
    assert_almost_equal(read_values, values, rtol=rtol, atol=0.0)
    if fast_reader and fast_reader.get('use_fast_converter'):
        fast_reader.update({'exponent_style': 'A'})
    else:
        pytest.skip('Fortran exponent style only available in fast converter')
    fields = ['.0101D202', '0.000000314d+314', '1777+305', '-1799E+305', '0.2e-323', '2500-327', ' 0.0000000000000000000001024Q+330']
    with ctx as w:
        t = ascii.read(StringIO(' '.join(fields)), format='no_header', guess=guess, fast_reader=fast_reader)
    if test_for_warnings:
        if sys.platform == 'win32' and (not fast_reader.get('use_fast_converter')):
            assert len(w) == 2
        else:
            assert len(w) == 3
    read_values = np.array([col[0] for col in t.itercols()])
    assert_almost_equal(read_values, values, rtol=rtol, atol=0.0)

@pytest.mark.parametrize('guess', [True, False])
@pytest.mark.parametrize('fast_reader', [False, {'use_fast_converter': False}, {'use_fast_converter': True}])
@pytest.mark.parametrize('parallel', [False, True])
def test_data_at_range_limit(parallel, fast_reader, guess):
    if False:
        i = 10
        return i + 15
    '\n    Test parsing of fixed-format float64 numbers near range limits\n    (|~4.94e-324 to 1.7977e+308|) - within limit for full precision\n    (|~2.5e-307| for strtod C parser, factor 10 better for fast_converter)\n    exact numbers shall be returned, beyond that an Overflow warning raised.\n    Input of exactly 0.0 must not raise an OverflowError.\n    '
    rtol = 1e-30
    if sys.platform == 'win32':
        ctx = nullcontext()
    else:
        ctx = pytest.warns()
    if fast_reader:
        fast_reader['parallel'] = parallel
        if fast_reader.get('use_fast_converter'):
            rtol = 1e-15
            ctx = pytest.warns()
        elif sys.maxsize < 2 ** 32:
            pytest.xfail('C parser cannot handle float64 on 32bit systems')
    if parallel:
        if not fast_reader:
            pytest.skip('Multiprocessing only available in fast reader')
        elif CI:
            pytest.xfail('Multiprocessing can sometimes fail on CI')
    for D in (99, 202, 305):
        t = ascii.read(StringIO(99 * '0' + '.' + D * '0' + '1'), format='no_header', guess=guess, fast_reader=fast_reader)
        assert_almost_equal(t['col1'][0], 10.0 ** (-(D + 1)), rtol=rtol, atol=0.0)
    for D in (99, 202, 308):
        t = ascii.read(StringIO('1' + D * '0' + '.0'), format='no_header', guess=guess, fast_reader=fast_reader)
        assert_almost_equal(t['col1'][0], 10.0 ** D, rtol=rtol, atol=0.0)
    for s in ('0.0', '0.0e+0', 399 * '0' + '.' + 365 * '0'):
        t = ascii.read(StringIO(s), format='no_header', guess=guess, fast_reader=fast_reader)
        assert t['col1'][0] == 0.0
    if parallel:
        pytest.skip('Catching warnings broken in parallel mode')
    elif not fast_reader:
        pytest.skip('Python/numpy reader does not raise on Overflow')
    with ctx as w:
        t = ascii.read(StringIO('0.' + 314 * '0' + '1'), format='no_header', guess=guess, fast_reader=fast_reader)
    if not isinstance(ctx, nullcontext):
        assert len(w) == 1, f'Expected 1 warning, found {len(w)}'
        assert 'OverflowError converting to FloatType in column col1, possibly resulting in degraded precision' in str(w[0].message)
    assert_almost_equal(t['col1'][0], 1e-315, rtol=1e-10, atol=0.0)

@pytest.mark.parametrize('guess', [True, False])
@pytest.mark.parametrize('parallel', [False, True])
def test_int_out_of_range(parallel, guess):
    if False:
        while True:
            i = 10
    "\n    Integer numbers outside int range shall be returned as string columns\n    consistent with the standard (Python) parser (no 'upcasting' to float).\n    "
    imin = np.iinfo(int).min + 1
    imax = np.iinfo(int).max - 1
    huge = f'{imax + 2:d}'
    text = f'P M S\n {imax:d} {imin:d} {huge:s}'
    expected = Table([[imax], [imin], [huge]], names=('P', 'M', 'S'))
    with pytest.warns() as w:
        table = ascii.read(text, format='basic', guess=guess, fast_reader={'parallel': parallel})
    if not parallel:
        assert len(w) == 1
        assert 'OverflowError converting to IntType in column S, reverting to String' in str(w[0].message)
    assert_table_equal(table, expected)
    text = f'P M S\n000{imax:d} -0{-imin:d} 00{huge:s}'
    expected = Table([[imax], [imin], ['00' + huge]], names=('P', 'M', 'S'))
    with pytest.warns() as w:
        table = ascii.read(text, format='basic', guess=guess, fast_reader={'parallel': parallel})
    if not parallel:
        assert len(w) == 1
        assert 'OverflowError converting to IntType in column S, reverting to String' in str(w[0].message)
    assert_table_equal(table, expected)

@pytest.mark.parametrize('guess', [True, False])
def test_int_out_of_order(guess):
    if False:
        print('Hello World!')
    '\n    Mixed columns should be returned as float, but if the out-of-range integer\n    shows up first, it will produce a string column - with both readers.\n    Broken with the parallel fast_reader.\n    '
    imax = np.iinfo(int).max - 1
    text = f'A B\n 12.3 {imax:d}0\n {imax:d}0 45.6e7'
    expected = Table([[12.3, 10.0 * imax], [f'{imax:d}0', '45.6e7']], names=('A', 'B'))
    with pytest.warns(AstropyWarning, match='OverflowError converting to IntType in column B, reverting to String'):
        table = ascii.read(text, format='basic', guess=guess, fast_reader=True)
    assert_table_equal(table, expected)
    with pytest.warns(AstropyWarning, match='OverflowError converting to IntType in column B, reverting to String'):
        table = ascii.read(text, format='basic', guess=guess, fast_reader=False)
    assert_table_equal(table, expected)

@pytest.mark.parametrize('guess', [True, False])
@pytest.mark.parametrize('parallel', [False, True])
def test_fortran_reader(parallel, guess):
    if False:
        while True:
            i = 10
    '\n    Make sure that ascii.read() can read Fortran-style exponential notation\n    using the fast_reader.\n    '
    rtol = 1e-15
    atol = 0.0
    text = 'A B C D\n100.01{:s}99       2.0  2.0{:s}-103 3\n 4.2{:s}-1 5.0{:s}-1     0.6{:s}4 .017{:s}+309'
    expc = Table([[1.0001e+101, 0.42], [2, 0.5], [2e-103, 6000.0], [3, 1.7e+307]], names=('A', 'B', 'C', 'D'))
    expstyles = {'e': 6 * 'E', 'D': ('D', 'd', 'd', 'D', 'd', 'D'), 'Q': 3 * ('q', 'Q'), 'Fortran': ('E', '0', 'D', 'Q', 'd', '0')}
    with pytest.raises(FastOptionsError) as e:
        ascii.read(text.format(*6 * 'D'), format='basic', guess=guess, fast_reader={'use_fast_converter': False, 'parallel': parallel, 'exponent_style': 'D'})
    assert 'fast_reader: exponent_style requires use_fast_converter' in str(e.value)
    for (s, c) in expstyles.items():
        table = ascii.read(text.format(*c), guess=guess, fast_reader={'parallel': parallel, 'exponent_style': s})
        assert_table_equal(table, expc, rtol=rtol, atol=atol)
    text = 'A B\t\t C D\n1.0001+101 2.0+000\t 0.0002-099 3\n 0.42-000 \t 0.5 6.+003   0.000000000000000000000017+330'
    table = ascii.read(text, guess=guess, fast_reader={'parallel': parallel, 'exponent_style': 'A'})
    assert_table_equal(table, expc, rtol=rtol, atol=atol)

@pytest.mark.parametrize('guess', [True, False])
@pytest.mark.parametrize('parallel', [False, True])
def test_fortran_invalid_exp(parallel, guess):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test Fortran-style exponential notation in the fast_reader with invalid\n    exponent-like patterns (no triple-digits) to make sure they are returned\n    as strings instead, as with the standard C parser.\n    '
    if parallel and CI:
        pytest.xfail('Multiprocessing can sometimes fail on CI')
    formats = {'basic': ' ', 'tab': '\t', 'csv': ','}
    header = ['S1', 'F2', 'S2', 'F3', 'S3', 'F4', 'F5', 'S4', 'I1', 'F6', 'F7']
    fields = ['1.0001+1', '.42d1', '2.3+10', '0.5', '3+1001', '3000.', '2', '4.56e-2.3', '8000', '4.2-022', '.00000145e314']
    vals_e = ['1.0001+1', '.42d1', '2.3+10', 0.5, '3+1001', 3000.0, 2, '4.56e-2.3', 8000, '4.2-022', 1.45e+308]
    vals_d = ['1.0001+1', 4.2, '2.3+10', 0.5, '3+1001', 3000.0, 2, '4.56e-2.3', 8000, '4.2-022', '.00000145e314']
    vals_a = ['1.0001+1', 4.2, '2.3+10', 0.5, '3+1001', 3000.0, 2, '4.56e-2.3', 8000, 4.2e-22, 1.45e+308]
    vals_v = ['1.0001+1', 4.2, '2.3+10', 0.5, '3+1001', 3000.0, 2, '4.56e-2.3', 8000, '4.2-022', 1.45e+308]
    for (f, s) in formats.items():
        t1 = ascii.read(StringIO(s.join(header) + '\n' + s.join(fields)), format=f, guess=guess, fast_reader={'parallel': parallel, 'exponent_style': 'A'})
        assert_table_equal(t1, Table([[col] for col in vals_a], names=header))
    if guess:
        formats['bar'] = '|'
    else:
        formats = {'basic': ' '}
    for s in formats.values():
        t2 = ascii.read(StringIO(s.join(header) + '\n' + s.join(fields)), guess=guess, fast_reader={'parallel': parallel, 'exponent_style': 'a'})
        assert_table_equal(t2, Table([[col] for col in vals_a], names=header))
    for s in formats.values():
        t3 = ascii.read(StringIO(s.join(header) + '\n' + s.join(fields)), guess=guess, fast_reader={'parallel': parallel, 'use_fast_converter': True})
        assert_table_equal(t3, Table([[col] for col in vals_e], names=header))
    for s in formats.values():
        t4 = ascii.read(StringIO(s.join(header) + '\n' + s.join(fields)), guess=guess, fast_reader={'parallel': parallel, 'exponent_style': 'D'})
        assert_table_equal(t4, Table([[col] for col in vals_d], names=header))
    for s in formats.values():
        t5 = ascii.read(StringIO(s.join(header) + '\n' + s.join(fields)), guess=guess, fast_reader={'parallel': parallel, 'use_fast_converter': False})
        read_values = [col[0] for col in t5.itercols()]
        if os.name == 'nt':
            assert read_values in (vals_v, vals_e)
        else:
            assert read_values == vals_e

def test_fortran_reader_notbasic():
    if False:
        while True:
            i = 10
    "\n    Check if readers without a fast option raise a value error when a\n    fast_reader is asked for (implies the default 'guess=True').\n    "
    tabstr = dedent('\n    a b\n    1 1.23D4\n    2 5.67D-8\n    ')[1:-1]
    t1 = ascii.read(tabstr.split('\n'), fast_reader={'exponent_style': 'D'})
    assert t1['b'].dtype.kind == 'f'
    tabrdb = dedent('\n    a\tb\n    # A simple RDB table\n    N\tN\n    1\t 1.23D4\n    2\t 5.67-008\n    ')[1:-1]
    t2 = ascii.read(tabrdb.split('\n'), format='rdb', fast_reader={'exponent_style': 'fortran'})
    assert t2['b'].dtype.kind == 'f'
    tabrst = dedent('\n    = =======\n    a b\n    = =======\n    1 1.23E4\n    2 5.67E-8\n    = =======\n    ')[1:-1]
    t3 = ascii.read(tabrst.split('\n'), format='rst')
    assert t3['b'].dtype.kind == 'f'
    t4 = ascii.read(tabrst.split('\n'), guess=True)
    assert t4['b'].dtype.kind == 'f'
    t5 = ascii.read(tabrst.split('\n'), format='rst', fast_reader=True)
    assert t5['b'].dtype.kind == 'f'
    with pytest.raises(ParameterError):
        ascii.read(tabrst.split('\n'), format='rst', guess=False, fast_reader='force')
    with pytest.raises(ParameterError):
        ascii.read(tabrst.split('\n'), format='rst', guess=False, fast_reader={'use_fast_converter': False})
    tabrst = tabrst.replace('E', 'D')
    with pytest.raises(ParameterError):
        ascii.read(tabrst.split('\n'), format='rst', guess=False, fast_reader={'exponent_style': 'D'})

@pytest.mark.parametrize('guess', [True, False])
@pytest.mark.parametrize('fast_reader', [{'exponent_style': 'D'}, {'exponent_style': 'A'}])
def test_dict_kwarg_integrity(fast_reader, guess):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check if dictionaries passed as kwargs (fast_reader in this test) are\n    left intact by ascii.read()\n    '
    expstyle = fast_reader.get('exponent_style', 'E')
    fields = ['10.1D+199', '3.14d+313', '2048d+306', '0.6D-325', '-2.d345']
    ascii.read(StringIO(' '.join(fields)), guess=guess, fast_reader=fast_reader)
    assert fast_reader.get('exponent_style', None) == expstyle

@pytest.mark.parametrize('fast_reader', [False, {'parallel': True}, {'parallel': False}])
def test_read_empty_basic_table_with_comments(fast_reader):
    if False:
        while True:
            i = 10
    '\n    Test for reading a "basic" format table that has no data but has comments.\n    Tests the fix for #8267.\n    '
    dat = '\n    # comment 1\n    # comment 2\n    col1 col2\n    '
    t = ascii.read(dat, fast_reader=fast_reader)
    assert t.meta['comments'] == ['comment 1', 'comment 2']
    assert len(t) == 0
    assert t.colnames == ['col1', 'col2']

@pytest.mark.parametrize('fast_reader', [{'use_fast_converter': True}, {'exponent_style': 'A'}])
def test_conversion_fast(fast_reader):
    if False:
        while True:
            i = 10
    '\n    The reader should try to convert each column to ints. If this fails, the\n    reader should try to convert to floats. Failing this, i.e. on parsing\n    non-numeric input including isolated positive/negative signs, it should\n    fall back to strings.\n    '
    text = '\n    A B C D E F G H\n    1 a 3 4 5 6 7 8\n    2. 1 9 -.1e1 10.0 8.7 6 -5.3e4\n    4 2 -12 .4 +.e1 - + six\n    '
    table = ascii.read(text, fast_reader=fast_reader)
    assert_equal(table['A'].dtype.kind, 'f')
    assert table['B'].dtype.kind in ('S', 'U')
    assert_equal(table['C'].dtype.kind, 'i')
    assert_equal(table['D'].dtype.kind, 'f')
    assert table['E'].dtype.kind in ('S', 'U')
    assert table['F'].dtype.kind in ('S', 'U')
    assert table['G'].dtype.kind in ('S', 'U')
    assert table['H'].dtype.kind in ('S', 'U')

@pytest.mark.parametrize('delimiter', ['\n', '\r'])
@pytest.mark.parametrize('fast_reader', [False, True, 'force'])
def test_newline_as_delimiter(delimiter, fast_reader):
    if False:
        print('Hello World!')
    '\n    Check that newline characters are correctly handled as delimiters.\n    Tests the fix for #9928.\n    '
    if delimiter == '\r':
        eol = '\n'
    else:
        eol = '\r'
    inp0 = ['a  | b | c ', " 1 | '2' | 3.00000 "]
    inp1 = "a {0:s} b {0:s}c{1:s} 1 {0:s}'2'{0:s} 3.0".format(delimiter, eol)
    inp2 = [f'a {delimiter} b{delimiter} c', f"1{delimiter} '2' {delimiter} 3.0"]
    t0 = ascii.read(inp0, delimiter='|', fast_reader=fast_reader)
    t1 = ascii.read(inp1, delimiter=delimiter, fast_reader=fast_reader)
    t2 = ascii.read(inp2, delimiter=delimiter, fast_reader=fast_reader)
    assert t1.colnames == t2.colnames == ['a', 'b', 'c']
    assert len(t1) == len(t2) == 1
    assert t1['b'].dtype.kind in ('S', 'U')
    assert t2['b'].dtype.kind in ('S', 'U')
    assert_table_equal(t1, t0)
    assert_table_equal(t2, t0)
    inp0 = 'a {0:s} b {0:s} c{1:s} 1 {0:s}"2"{0:s} 3.0'.format('|', eol)
    inp1 = 'a {0:s} b {0:s} c{1:s} 1 {0:s}"2"{0:s} 3.0'.format(delimiter, eol)
    t0 = ascii.read(inp0, delimiter='|', fast_reader=fast_reader)
    t1 = ascii.read(inp1, delimiter=delimiter, fast_reader=fast_reader)
    if not fast_reader:
        pytest.xfail('Quoted fields are not parsed correctly by BaseSplitter')
    assert_equal(t1['b'].dtype.kind, 'i')

@pytest.mark.parametrize('delimiter', [' ', '|', '\n', '\r'])
@pytest.mark.parametrize('fast_reader', [False, True, 'force'])
def test_single_line_string(delimiter, fast_reader):
    if False:
        return 10
    '\n    String input without a newline character is interpreted as filename,\n    unless element of an iterable. Maybe not logical, but test that it is\n    at least treated consistently.\n    '
    expected = Table([[1], [2], [3.0]], names=('col1', 'col2', 'col3'))
    text = f'1{delimiter:s}2{delimiter:s}3.0'
    if delimiter in ('\r', '\n'):
        t1 = ascii.read(text, format='no_header', delimiter=delimiter, fast_reader=fast_reader)
        assert_table_equal(t1, expected)
    else:
        with pytest.raises((FileNotFoundError, OSError)):
            t1 = ascii.read(text, format='no_header', delimiter=delimiter, fast_reader=fast_reader)
    t2 = ascii.read([text], format='no_header', delimiter=delimiter, fast_reader=fast_reader)
    assert_table_equal(t2, expected)