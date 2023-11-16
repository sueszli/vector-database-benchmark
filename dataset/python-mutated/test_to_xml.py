from __future__ import annotations
from io import BytesIO, StringIO
import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import NA, DataFrame, Index
import pandas._testing as tm
from pandas.io.common import get_handle
from pandas.io.xml import read_xml

@pytest.fixture
def geom_df():
    if False:
        return 10
    return DataFrame({'shape': ['square', 'circle', 'triangle'], 'degrees': [360, 360, 180], 'sides': [4, np.nan, 3]})

@pytest.fixture
def planet_df():
    if False:
        print('Hello World!')
    return DataFrame({'planet': ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'], 'type': ['terrestrial', 'terrestrial', 'terrestrial', 'terrestrial', 'gas giant', 'gas giant', 'ice giant', 'ice giant'], 'location': ['inner', 'inner', 'inner', 'inner', 'outer', 'outer', 'outer', 'outer'], 'mass': [0.330114, 4.86747, 5.97237, 0.641712, 1898.187, 568.3174, 86.8127, 102.4126]})

@pytest.fixture
def from_file_expected():
    if False:
        print('Hello World!')
    return "<?xml version='1.0' encoding='utf-8'?>\n<data>\n  <row>\n    <index>0</index>\n    <category>cooking</category>\n    <title>Everyday Italian</title>\n    <author>Giada De Laurentiis</author>\n    <year>2005</year>\n    <price>30.0</price>\n  </row>\n  <row>\n    <index>1</index>\n    <category>children</category>\n    <title>Harry Potter</title>\n    <author>J K. Rowling</author>\n    <year>2005</year>\n    <price>29.99</price>\n  </row>\n  <row>\n    <index>2</index>\n    <category>web</category>\n    <title>Learning XML</title>\n    <author>Erik T. Ray</author>\n    <year>2003</year>\n    <price>39.95</price>\n  </row>\n</data>"

def equalize_decl(doc):
    if False:
        for i in range(10):
            print('nop')
    if doc is not None:
        doc = doc.replace('<?xml version="1.0" encoding="utf-8"?', "<?xml version='1.0' encoding='utf-8'?")
    return doc

@pytest.fixture(params=['rb', 'r'])
def mode(request):
    if False:
        return 10
    return request.param

@pytest.fixture(params=[pytest.param('lxml', marks=td.skip_if_no('lxml')), 'etree'])
def parser(request):
    if False:
        i = 10
        return i + 15
    return request.param

def test_file_output_str_read(xml_books, parser, from_file_expected):
    if False:
        i = 10
        return i + 15
    df_file = read_xml(xml_books, parser=parser)
    with tm.ensure_clean('test.xml') as path:
        df_file.to_xml(path, parser=parser)
        with open(path, 'rb') as f:
            output = f.read().decode('utf-8').strip()
        output = equalize_decl(output)
        assert output == from_file_expected

def test_file_output_bytes_read(xml_books, parser, from_file_expected):
    if False:
        i = 10
        return i + 15
    df_file = read_xml(xml_books, parser=parser)
    with tm.ensure_clean('test.xml') as path:
        df_file.to_xml(path, parser=parser)
        with open(path, 'rb') as f:
            output = f.read().decode('utf-8').strip()
        output = equalize_decl(output)
        assert output == from_file_expected

def test_str_output(xml_books, parser, from_file_expected):
    if False:
        return 10
    df_file = read_xml(xml_books, parser=parser)
    output = df_file.to_xml(parser=parser)
    output = equalize_decl(output)
    assert output == from_file_expected

def test_wrong_file_path(parser, geom_df):
    if False:
        print('Hello World!')
    path = '/my/fake/path/output.xml'
    with pytest.raises(OSError, match='Cannot save file into a non-existent directory: .*path'):
        geom_df.to_xml(path, parser=parser)

def test_index_false(xml_books, parser):
    if False:
        i = 10
        return i + 15
    expected = "<?xml version='1.0' encoding='utf-8'?>\n<data>\n  <row>\n    <category>cooking</category>\n    <title>Everyday Italian</title>\n    <author>Giada De Laurentiis</author>\n    <year>2005</year>\n    <price>30.0</price>\n  </row>\n  <row>\n    <category>children</category>\n    <title>Harry Potter</title>\n    <author>J K. Rowling</author>\n    <year>2005</year>\n    <price>29.99</price>\n  </row>\n  <row>\n    <category>web</category>\n    <title>Learning XML</title>\n    <author>Erik T. Ray</author>\n    <year>2003</year>\n    <price>39.95</price>\n  </row>\n</data>"
    df_file = read_xml(xml_books, parser=parser)
    with tm.ensure_clean('test.xml') as path:
        df_file.to_xml(path, index=False, parser=parser)
        with open(path, 'rb') as f:
            output = f.read().decode('utf-8').strip()
        output = equalize_decl(output)
        assert output == expected

def test_index_false_rename_row_root(xml_books, parser):
    if False:
        while True:
            i = 10
    expected = "<?xml version='1.0' encoding='utf-8'?>\n<books>\n  <book>\n    <category>cooking</category>\n    <title>Everyday Italian</title>\n    <author>Giada De Laurentiis</author>\n    <year>2005</year>\n    <price>30.0</price>\n  </book>\n  <book>\n    <category>children</category>\n    <title>Harry Potter</title>\n    <author>J K. Rowling</author>\n    <year>2005</year>\n    <price>29.99</price>\n  </book>\n  <book>\n    <category>web</category>\n    <title>Learning XML</title>\n    <author>Erik T. Ray</author>\n    <year>2003</year>\n    <price>39.95</price>\n  </book>\n</books>"
    df_file = read_xml(xml_books, parser=parser)
    with tm.ensure_clean('test.xml') as path:
        df_file.to_xml(path, index=False, root_name='books', row_name='book', parser=parser)
        with open(path, 'rb') as f:
            output = f.read().decode('utf-8').strip()
        output = equalize_decl(output)
        assert output == expected

@pytest.mark.parametrize('offset_index', [list(range(10, 13)), [str(i) for i in range(10, 13)]])
def test_index_false_with_offset_input_index(parser, offset_index, geom_df):
    if False:
        print('Hello World!')
    '\n    Tests that the output does not contain the `<index>` field when the index of the\n    input Dataframe has an offset.\n\n    This is a regression test for issue #42458.\n    '
    expected = "<?xml version='1.0' encoding='utf-8'?>\n<data>\n  <row>\n    <shape>square</shape>\n    <degrees>360</degrees>\n    <sides>4.0</sides>\n  </row>\n  <row>\n    <shape>circle</shape>\n    <degrees>360</degrees>\n    <sides/>\n  </row>\n  <row>\n    <shape>triangle</shape>\n    <degrees>180</degrees>\n    <sides>3.0</sides>\n  </row>\n</data>"
    offset_geom_df = geom_df.copy()
    offset_geom_df.index = Index(offset_index)
    output = offset_geom_df.to_xml(index=False, parser=parser)
    output = equalize_decl(output)
    assert output == expected
na_expected = "<?xml version='1.0' encoding='utf-8'?>\n<data>\n  <row>\n    <index>0</index>\n    <shape>square</shape>\n    <degrees>360</degrees>\n    <sides>4.0</sides>\n  </row>\n  <row>\n    <index>1</index>\n    <shape>circle</shape>\n    <degrees>360</degrees>\n    <sides/>\n  </row>\n  <row>\n    <index>2</index>\n    <shape>triangle</shape>\n    <degrees>180</degrees>\n    <sides>3.0</sides>\n  </row>\n</data>"

def test_na_elem_output(parser, geom_df):
    if False:
        return 10
    output = geom_df.to_xml(parser=parser)
    output = equalize_decl(output)
    assert output == na_expected

def test_na_empty_str_elem_option(parser, geom_df):
    if False:
        for i in range(10):
            print('nop')
    output = geom_df.to_xml(na_rep='', parser=parser)
    output = equalize_decl(output)
    assert output == na_expected

def test_na_empty_elem_option(parser, geom_df):
    if False:
        i = 10
        return i + 15
    expected = "<?xml version='1.0' encoding='utf-8'?>\n<data>\n  <row>\n    <index>0</index>\n    <shape>square</shape>\n    <degrees>360</degrees>\n    <sides>4.0</sides>\n  </row>\n  <row>\n    <index>1</index>\n    <shape>circle</shape>\n    <degrees>360</degrees>\n    <sides>0.0</sides>\n  </row>\n  <row>\n    <index>2</index>\n    <shape>triangle</shape>\n    <degrees>180</degrees>\n    <sides>3.0</sides>\n  </row>\n</data>"
    output = geom_df.to_xml(na_rep='0.0', parser=parser)
    output = equalize_decl(output)
    assert output == expected

def test_attrs_cols_nan_output(parser, geom_df):
    if False:
        for i in range(10):
            print('nop')
    expected = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<data>\n  <row index="0" shape="square" degrees="360" sides="4.0"/>\n  <row index="1" shape="circle" degrees="360"/>\n  <row index="2" shape="triangle" degrees="180" sides="3.0"/>\n</data>'
    output = geom_df.to_xml(attr_cols=['shape', 'degrees', 'sides'], parser=parser)
    output = equalize_decl(output)
    assert output == expected

def test_attrs_cols_prefix(parser, geom_df):
    if False:
        print('Hello World!')
    expected = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<doc:data xmlns:doc="http://example.xom">\n  <doc:row doc:index="0" doc:shape="square" doc:degrees="360" doc:sides="4.0"/>\n  <doc:row doc:index="1" doc:shape="circle" doc:degrees="360"/>\n  <doc:row doc:index="2" doc:shape="triangle" doc:degrees="180" doc:sides="3.0"/>\n</doc:data>'
    output = geom_df.to_xml(attr_cols=['index', 'shape', 'degrees', 'sides'], namespaces={'doc': 'http://example.xom'}, prefix='doc', parser=parser)
    output = equalize_decl(output)
    assert output == expected

def test_attrs_unknown_column(parser, geom_df):
    if False:
        while True:
            i = 10
    with pytest.raises(KeyError, match='no valid column'):
        geom_df.to_xml(attr_cols=['shape', 'degree', 'sides'], parser=parser)

def test_attrs_wrong_type(parser, geom_df):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(TypeError, match='is not a valid type for attr_cols'):
        geom_df.to_xml(attr_cols='"shape", "degree", "sides"', parser=parser)

def test_elems_cols_nan_output(parser, geom_df):
    if False:
        return 10
    elems_cols_expected = "<?xml version='1.0' encoding='utf-8'?>\n<data>\n  <row>\n    <degrees>360</degrees>\n    <sides>4.0</sides>\n    <shape>square</shape>\n  </row>\n  <row>\n    <degrees>360</degrees>\n    <sides/>\n    <shape>circle</shape>\n  </row>\n  <row>\n    <degrees>180</degrees>\n    <sides>3.0</sides>\n    <shape>triangle</shape>\n  </row>\n</data>"
    output = geom_df.to_xml(index=False, elem_cols=['degrees', 'sides', 'shape'], parser=parser)
    output = equalize_decl(output)
    assert output == elems_cols_expected

def test_elems_unknown_column(parser, geom_df):
    if False:
        print('Hello World!')
    with pytest.raises(KeyError, match='no valid column'):
        geom_df.to_xml(elem_cols=['shape', 'degree', 'sides'], parser=parser)

def test_elems_wrong_type(parser, geom_df):
    if False:
        while True:
            i = 10
    with pytest.raises(TypeError, match='is not a valid type for elem_cols'):
        geom_df.to_xml(elem_cols='"shape", "degree", "sides"', parser=parser)

def test_elems_and_attrs_cols(parser, geom_df):
    if False:
        i = 10
        return i + 15
    elems_cols_expected = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<data>\n  <row shape="square">\n    <degrees>360</degrees>\n    <sides>4.0</sides>\n  </row>\n  <row shape="circle">\n    <degrees>360</degrees>\n    <sides/>\n  </row>\n  <row shape="triangle">\n    <degrees>180</degrees>\n    <sides>3.0</sides>\n  </row>\n</data>'
    output = geom_df.to_xml(index=False, elem_cols=['degrees', 'sides'], attr_cols=['shape'], parser=parser)
    output = equalize_decl(output)
    assert output == elems_cols_expected

def test_hierarchical_columns(parser, planet_df):
    if False:
        return 10
    expected = "<?xml version='1.0' encoding='utf-8'?>\n<data>\n  <row>\n    <location>inner</location>\n    <type>terrestrial</type>\n    <count_mass>4</count_mass>\n    <sum_mass>11.81</sum_mass>\n    <mean_mass>2.95</mean_mass>\n  </row>\n  <row>\n    <location>outer</location>\n    <type>gas giant</type>\n    <count_mass>2</count_mass>\n    <sum_mass>2466.5</sum_mass>\n    <mean_mass>1233.25</mean_mass>\n  </row>\n  <row>\n    <location>outer</location>\n    <type>ice giant</type>\n    <count_mass>2</count_mass>\n    <sum_mass>189.23</sum_mass>\n    <mean_mass>94.61</mean_mass>\n  </row>\n  <row>\n    <location>All</location>\n    <type/>\n    <count_mass>8</count_mass>\n    <sum_mass>2667.54</sum_mass>\n    <mean_mass>333.44</mean_mass>\n  </row>\n</data>"
    pvt = planet_df.pivot_table(index=['location', 'type'], values='mass', aggfunc=['count', 'sum', 'mean'], margins=True).round(2)
    output = pvt.to_xml(parser=parser)
    output = equalize_decl(output)
    assert output == expected

def test_hierarchical_attrs_columns(parser, planet_df):
    if False:
        while True:
            i = 10
    expected = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<data>\n  <row location="inner" type="terrestrial" count_mass="4" sum_mass="11.81" mean_mass="2.95"/>\n  <row location="outer" type="gas giant" count_mass="2" sum_mass="2466.5" mean_mass="1233.25"/>\n  <row location="outer" type="ice giant" count_mass="2" sum_mass="189.23" mean_mass="94.61"/>\n  <row location="All" type="" count_mass="8" sum_mass="2667.54" mean_mass="333.44"/>\n</data>'
    pvt = planet_df.pivot_table(index=['location', 'type'], values='mass', aggfunc=['count', 'sum', 'mean'], margins=True).round(2)
    output = pvt.to_xml(attr_cols=list(pvt.reset_index().columns.values), parser=parser)
    output = equalize_decl(output)
    assert output == expected

def test_multi_index(parser, planet_df):
    if False:
        i = 10
        return i + 15
    expected = "<?xml version='1.0' encoding='utf-8'?>\n<data>\n  <row>\n    <location>inner</location>\n    <type>terrestrial</type>\n    <count>4</count>\n    <sum>11.81</sum>\n    <mean>2.95</mean>\n  </row>\n  <row>\n    <location>outer</location>\n    <type>gas giant</type>\n    <count>2</count>\n    <sum>2466.5</sum>\n    <mean>1233.25</mean>\n  </row>\n  <row>\n    <location>outer</location>\n    <type>ice giant</type>\n    <count>2</count>\n    <sum>189.23</sum>\n    <mean>94.61</mean>\n  </row>\n</data>"
    agg = planet_df.groupby(['location', 'type'])['mass'].agg(['count', 'sum', 'mean']).round(2)
    output = agg.to_xml(parser=parser)
    output = equalize_decl(output)
    assert output == expected

def test_multi_index_attrs_cols(parser, planet_df):
    if False:
        return 10
    expected = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<data>\n  <row location="inner" type="terrestrial" count="4" sum="11.81" mean="2.95"/>\n  <row location="outer" type="gas giant" count="2" sum="2466.5" mean="1233.25"/>\n  <row location="outer" type="ice giant" count="2" sum="189.23" mean="94.61"/>\n</data>'
    agg = planet_df.groupby(['location', 'type'])['mass'].agg(['count', 'sum', 'mean']).round(2)
    output = agg.to_xml(attr_cols=list(agg.reset_index().columns.values), parser=parser)
    output = equalize_decl(output)
    assert output == expected

def test_default_namespace(parser, geom_df):
    if False:
        for i in range(10):
            print('nop')
    expected = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<data xmlns="http://example.com">\n  <row>\n    <index>0</index>\n    <shape>square</shape>\n    <degrees>360</degrees>\n    <sides>4.0</sides>\n  </row>\n  <row>\n    <index>1</index>\n    <shape>circle</shape>\n    <degrees>360</degrees>\n    <sides/>\n  </row>\n  <row>\n    <index>2</index>\n    <shape>triangle</shape>\n    <degrees>180</degrees>\n    <sides>3.0</sides>\n  </row>\n</data>'
    output = geom_df.to_xml(namespaces={'': 'http://example.com'}, parser=parser)
    output = equalize_decl(output)
    assert output == expected

def test_unused_namespaces(parser, geom_df):
    if False:
        return 10
    expected = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<data xmlns:oth="http://other.org" xmlns:ex="http://example.com">\n  <row>\n    <index>0</index>\n    <shape>square</shape>\n    <degrees>360</degrees>\n    <sides>4.0</sides>\n  </row>\n  <row>\n    <index>1</index>\n    <shape>circle</shape>\n    <degrees>360</degrees>\n    <sides/>\n  </row>\n  <row>\n    <index>2</index>\n    <shape>triangle</shape>\n    <degrees>180</degrees>\n    <sides>3.0</sides>\n  </row>\n</data>'
    output = geom_df.to_xml(namespaces={'oth': 'http://other.org', 'ex': 'http://example.com'}, parser=parser)
    output = equalize_decl(output)
    assert output == expected

def test_namespace_prefix(parser, geom_df):
    if False:
        print('Hello World!')
    expected = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<doc:data xmlns:doc="http://example.com">\n  <doc:row>\n    <doc:index>0</doc:index>\n    <doc:shape>square</doc:shape>\n    <doc:degrees>360</doc:degrees>\n    <doc:sides>4.0</doc:sides>\n  </doc:row>\n  <doc:row>\n    <doc:index>1</doc:index>\n    <doc:shape>circle</doc:shape>\n    <doc:degrees>360</doc:degrees>\n    <doc:sides/>\n  </doc:row>\n  <doc:row>\n    <doc:index>2</doc:index>\n    <doc:shape>triangle</doc:shape>\n    <doc:degrees>180</doc:degrees>\n    <doc:sides>3.0</doc:sides>\n  </doc:row>\n</doc:data>'
    output = geom_df.to_xml(namespaces={'doc': 'http://example.com'}, prefix='doc', parser=parser)
    output = equalize_decl(output)
    assert output == expected

def test_missing_prefix_in_nmsp(parser, geom_df):
    if False:
        i = 10
        return i + 15
    with pytest.raises(KeyError, match='doc is not included in namespaces'):
        geom_df.to_xml(namespaces={'': 'http://example.com'}, prefix='doc', parser=parser)

def test_namespace_prefix_and_default(parser, geom_df):
    if False:
        i = 10
        return i + 15
    expected = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<doc:data xmlns:doc="http://other.org" xmlns="http://example.com">\n  <doc:row>\n    <doc:index>0</doc:index>\n    <doc:shape>square</doc:shape>\n    <doc:degrees>360</doc:degrees>\n    <doc:sides>4.0</doc:sides>\n  </doc:row>\n  <doc:row>\n    <doc:index>1</doc:index>\n    <doc:shape>circle</doc:shape>\n    <doc:degrees>360</doc:degrees>\n    <doc:sides/>\n  </doc:row>\n  <doc:row>\n    <doc:index>2</doc:index>\n    <doc:shape>triangle</doc:shape>\n    <doc:degrees>180</doc:degrees>\n    <doc:sides>3.0</doc:sides>\n  </doc:row>\n</doc:data>'
    output = geom_df.to_xml(namespaces={'': 'http://example.com', 'doc': 'http://other.org'}, prefix='doc', parser=parser)
    output = equalize_decl(output)
    assert output == expected
encoding_expected = "<?xml version='1.0' encoding='ISO-8859-1'?>\n<data>\n  <row>\n    <index>0</index>\n    <rank>1</rank>\n    <malename>José</malename>\n    <femalename>Sofía</femalename>\n  </row>\n  <row>\n    <index>1</index>\n    <rank>2</rank>\n    <malename>Luis</malename>\n    <femalename>Valentina</femalename>\n  </row>\n  <row>\n    <index>2</index>\n    <rank>3</rank>\n    <malename>Carlos</malename>\n    <femalename>Isabella</femalename>\n  </row>\n  <row>\n    <index>3</index>\n    <rank>4</rank>\n    <malename>Juan</malename>\n    <femalename>Camila</femalename>\n  </row>\n  <row>\n    <index>4</index>\n    <rank>5</rank>\n    <malename>Jorge</malename>\n    <femalename>Valeria</femalename>\n  </row>\n</data>"

def test_encoding_option_str(xml_baby_names, parser):
    if False:
        i = 10
        return i + 15
    df_file = read_xml(xml_baby_names, parser=parser, encoding='ISO-8859-1').head(5)
    output = df_file.to_xml(encoding='ISO-8859-1', parser=parser)
    if output is not None:
        output = output.replace('<?xml version="1.0" encoding="ISO-8859-1"?', "<?xml version='1.0' encoding='ISO-8859-1'?")
    assert output == encoding_expected

def test_correct_encoding_file(xml_baby_names):
    if False:
        print('Hello World!')
    pytest.importorskip('lxml')
    df_file = read_xml(xml_baby_names, encoding='ISO-8859-1', parser='lxml')
    with tm.ensure_clean('test.xml') as path:
        df_file.to_xml(path, index=False, encoding='ISO-8859-1', parser='lxml')

@pytest.mark.parametrize('encoding', ['UTF-8', 'UTF-16', 'ISO-8859-1'])
def test_wrong_encoding_option_lxml(xml_baby_names, parser, encoding):
    if False:
        for i in range(10):
            print('nop')
    pytest.importorskip('lxml')
    df_file = read_xml(xml_baby_names, encoding='ISO-8859-1', parser='lxml')
    with tm.ensure_clean('test.xml') as path:
        df_file.to_xml(path, index=False, encoding=encoding, parser=parser)

def test_misspelled_encoding(parser, geom_df):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(LookupError, match='unknown encoding'):
        geom_df.to_xml(encoding='uft-8', parser=parser)

def test_xml_declaration_pretty_print(geom_df):
    if False:
        print('Hello World!')
    pytest.importorskip('lxml')
    expected = '<data>\n  <row>\n    <index>0</index>\n    <shape>square</shape>\n    <degrees>360</degrees>\n    <sides>4.0</sides>\n  </row>\n  <row>\n    <index>1</index>\n    <shape>circle</shape>\n    <degrees>360</degrees>\n    <sides/>\n  </row>\n  <row>\n    <index>2</index>\n    <shape>triangle</shape>\n    <degrees>180</degrees>\n    <sides>3.0</sides>\n  </row>\n</data>'
    output = geom_df.to_xml(xml_declaration=False)
    assert output == expected

def test_no_pretty_print_with_decl(parser, geom_df):
    if False:
        print('Hello World!')
    expected = "<?xml version='1.0' encoding='utf-8'?>\n<data><row><index>0</index><shape>square</shape><degrees>360</degrees><sides>4.0</sides></row><row><index>1</index><shape>circle</shape><degrees>360</degrees><sides/></row><row><index>2</index><shape>triangle</shape><degrees>180</degrees><sides>3.0</sides></row></data>"
    output = geom_df.to_xml(pretty_print=False, parser=parser)
    output = equalize_decl(output)
    if output is not None:
        output = output.replace(' />', '/>')
    assert output == expected

def test_no_pretty_print_no_decl(parser, geom_df):
    if False:
        for i in range(10):
            print('nop')
    expected = '<data><row><index>0</index><shape>square</shape><degrees>360</degrees><sides>4.0</sides></row><row><index>1</index><shape>circle</shape><degrees>360</degrees><sides/></row><row><index>2</index><shape>triangle</shape><degrees>180</degrees><sides>3.0</sides></row></data>'
    output = geom_df.to_xml(xml_declaration=False, pretty_print=False, parser=parser)
    if output is not None:
        output = output.replace(' />', '/>')
    assert output == expected

@td.skip_if_installed('lxml')
def test_default_parser_no_lxml(geom_df):
    if False:
        print('Hello World!')
    with pytest.raises(ImportError, match='lxml not found, please install or use the etree parser.'):
        geom_df.to_xml()

def test_unknown_parser(geom_df):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError, match='Values for parser can only be lxml or etree.'):
        geom_df.to_xml(parser='bs4')
xsl_expected = '<?xml version="1.0" encoding="utf-8"?>\n<data>\n  <row>\n    <field field="index">0</field>\n    <field field="shape">square</field>\n    <field field="degrees">360</field>\n    <field field="sides">4.0</field>\n  </row>\n  <row>\n    <field field="index">1</field>\n    <field field="shape">circle</field>\n    <field field="degrees">360</field>\n    <field field="sides"/>\n  </row>\n  <row>\n    <field field="index">2</field>\n    <field field="shape">triangle</field>\n    <field field="degrees">180</field>\n    <field field="sides">3.0</field>\n  </row>\n</data>'

def test_stylesheet_file_like(xsl_row_field_output, mode, geom_df):
    if False:
        print('Hello World!')
    pytest.importorskip('lxml')
    with open(xsl_row_field_output, mode, encoding='utf-8' if mode == 'r' else None) as f:
        assert geom_df.to_xml(stylesheet=f) == xsl_expected

def test_stylesheet_io(xsl_row_field_output, mode, geom_df):
    if False:
        while True:
            i = 10
    pytest.importorskip('lxml')
    xsl_obj: BytesIO | StringIO
    with open(xsl_row_field_output, mode, encoding='utf-8' if mode == 'r' else None) as f:
        if mode == 'rb':
            xsl_obj = BytesIO(f.read())
        else:
            xsl_obj = StringIO(f.read())
    output = geom_df.to_xml(stylesheet=xsl_obj)
    assert output == xsl_expected

def test_stylesheet_buffered_reader(xsl_row_field_output, mode, geom_df):
    if False:
        return 10
    pytest.importorskip('lxml')
    with open(xsl_row_field_output, mode, encoding='utf-8' if mode == 'r' else None) as f:
        xsl_obj = f.read()
    output = geom_df.to_xml(stylesheet=xsl_obj)
    assert output == xsl_expected

def test_stylesheet_wrong_path(geom_df):
    if False:
        return 10
    lxml_etree = pytest.importorskip('lxml.etree')
    xsl = os.path.join('data', 'xml', 'row_field_output.xslt')
    with pytest.raises(lxml_etree.XMLSyntaxError, match="Start tag expected, '<' not found"):
        geom_df.to_xml(stylesheet=xsl)

@pytest.mark.parametrize('val', ['', b''])
def test_empty_string_stylesheet(val, geom_df):
    if False:
        while True:
            i = 10
    lxml_etree = pytest.importorskip('lxml.etree')
    msg = '|'.join(['Document is empty', "Start tag expected, '<' not found", 'None \\(line 0\\)'])
    with pytest.raises(lxml_etree.XMLSyntaxError, match=msg):
        geom_df.to_xml(stylesheet=val)

def test_incorrect_xsl_syntax(geom_df):
    if False:
        return 10
    lxml_etree = pytest.importorskip('lxml.etree')
    xsl = '<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">\n    <xsl:output method="xml" encoding="utf-8" indent="yes" >\n    <xsl:strip-space elements="*"/>\n\n    <xsl:template match="@*|node()">\n        <xsl:copy>\n            <xsl:apply-templates select="@*|node()"/>\n        </xsl:copy>\n    </xsl:template>\n\n    <xsl:template match="row/*">\n        <field>\n            <xsl:attribute name="field">\n                <xsl:value-of select="name()"/>\n            </xsl:attribute>\n            <xsl:value-of select="text()"/>\n        </field>\n    </xsl:template>\n</xsl:stylesheet>'
    with pytest.raises(lxml_etree.XMLSyntaxError, match='Opening and ending tag mismatch'):
        geom_df.to_xml(stylesheet=xsl)

def test_incorrect_xsl_eval(geom_df):
    if False:
        while True:
            i = 10
    lxml_etree = pytest.importorskip('lxml.etree')
    xsl = '<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">\n    <xsl:output method="xml" encoding="utf-8" indent="yes" />\n    <xsl:strip-space elements="*"/>\n\n    <xsl:template match="@*|node(*)">\n        <xsl:copy>\n            <xsl:apply-templates select="@*|node()"/>\n        </xsl:copy>\n    </xsl:template>\n\n    <xsl:template match="row/*">\n        <field>\n            <xsl:attribute name="field">\n                <xsl:value-of select="name()"/>\n            </xsl:attribute>\n            <xsl:value-of select="text()"/>\n        </field>\n    </xsl:template>\n</xsl:stylesheet>'
    with pytest.raises(lxml_etree.XSLTParseError, match='failed to compile'):
        geom_df.to_xml(stylesheet=xsl)

def test_incorrect_xsl_apply(geom_df):
    if False:
        i = 10
        return i + 15
    lxml_etree = pytest.importorskip('lxml.etree')
    xsl = '<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">\n    <xsl:output method="xml" encoding="utf-8" indent="yes" />\n    <xsl:strip-space elements="*"/>\n\n    <xsl:template match="@*|node()">\n        <xsl:copy>\n            <xsl:copy-of select="document(\'non_existent.xml\')/*"/>\n        </xsl:copy>\n    </xsl:template>\n</xsl:stylesheet>'
    with pytest.raises(lxml_etree.XSLTApplyError, match='Cannot resolve URI'):
        with tm.ensure_clean('test.xml') as path:
            geom_df.to_xml(path, stylesheet=xsl)

def test_stylesheet_with_etree(geom_df):
    if False:
        i = 10
        return i + 15
    xsl = '<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">\n    <xsl:output method="xml" encoding="utf-8" indent="yes" />\n    <xsl:strip-space elements="*"/>\n\n    <xsl:template match="@*|node(*)">\n        <xsl:copy>\n            <xsl:apply-templates select="@*|node()"/>\n        </xsl:copy>\n    </xsl:template>'
    with pytest.raises(ValueError, match='To use stylesheet, you need lxml installed'):
        geom_df.to_xml(parser='etree', stylesheet=xsl)

def test_style_to_csv(geom_df):
    if False:
        return 10
    pytest.importorskip('lxml')
    xsl = '<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">\n    <xsl:output method="text" indent="yes" />\n    <xsl:strip-space elements="*"/>\n\n    <xsl:param name="delim">,</xsl:param>\n    <xsl:template match="/data">\n        <xsl:text>,shape,degrees,sides&#xa;</xsl:text>\n        <xsl:apply-templates select="row"/>\n    </xsl:template>\n\n    <xsl:template match="row">\n        <xsl:value-of select="concat(index, $delim, shape, $delim,\n                                     degrees, $delim, sides)"/>\n         <xsl:text>&#xa;</xsl:text>\n    </xsl:template>\n</xsl:stylesheet>'
    out_csv = geom_df.to_csv(lineterminator='\n')
    if out_csv is not None:
        out_csv = out_csv.strip()
    out_xml = geom_df.to_xml(stylesheet=xsl)
    assert out_csv == out_xml

def test_style_to_string(geom_df):
    if False:
        for i in range(10):
            print('nop')
    pytest.importorskip('lxml')
    xsl = '<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">\n    <xsl:output method="text" indent="yes" />\n    <xsl:strip-space elements="*"/>\n\n    <xsl:param name="delim"><xsl:text>               </xsl:text></xsl:param>\n    <xsl:template match="/data">\n        <xsl:text>      shape  degrees  sides&#xa;</xsl:text>\n        <xsl:apply-templates select="row"/>\n    </xsl:template>\n\n    <xsl:template match="row">\n        <xsl:value-of select="concat(index, \' \',\n                                     substring($delim, 1, string-length(\'triangle\')\n                                               - string-length(shape) + 1),\n                                     shape,\n                                     substring($delim, 1, string-length(name(degrees))\n                                               - string-length(degrees) + 2),\n                                     degrees,\n                                     substring($delim, 1, string-length(name(sides))\n                                               - string-length(sides) + 2),\n                                     sides)"/>\n         <xsl:text>&#xa;</xsl:text>\n    </xsl:template>\n</xsl:stylesheet>'
    out_str = geom_df.to_string()
    out_xml = geom_df.to_xml(na_rep='NaN', stylesheet=xsl)
    assert out_xml == out_str

def test_style_to_json(geom_df):
    if False:
        for i in range(10):
            print('nop')
    pytest.importorskip('lxml')
    xsl = '<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">\n    <xsl:output method="text" indent="yes" />\n    <xsl:strip-space elements="*"/>\n\n    <xsl:param name="quot">"</xsl:param>\n\n    <xsl:template match="/data">\n        <xsl:text>{"shape":{</xsl:text>\n        <xsl:apply-templates select="descendant::row/shape"/>\n        <xsl:text>},"degrees":{</xsl:text>\n        <xsl:apply-templates select="descendant::row/degrees"/>\n        <xsl:text>},"sides":{</xsl:text>\n        <xsl:apply-templates select="descendant::row/sides"/>\n        <xsl:text>}}</xsl:text>\n    </xsl:template>\n\n    <xsl:template match="shape|degrees|sides">\n        <xsl:variable name="val">\n            <xsl:if test = ".=\'\'">\n                <xsl:value-of select="\'null\'"/>\n            </xsl:if>\n            <xsl:if test = "number(text()) = text()">\n                <xsl:value-of select="text()"/>\n            </xsl:if>\n            <xsl:if test = "number(text()) != text()">\n                <xsl:value-of select="concat($quot, text(), $quot)"/>\n            </xsl:if>\n        </xsl:variable>\n        <xsl:value-of select="concat($quot, preceding-sibling::index,\n                                     $quot,\':\', $val)"/>\n        <xsl:if test="preceding-sibling::index != //row[last()]/index">\n            <xsl:text>,</xsl:text>\n        </xsl:if>\n    </xsl:template>\n</xsl:stylesheet>'
    out_json = geom_df.to_json()
    out_xml = geom_df.to_xml(stylesheet=xsl)
    assert out_json == out_xml
geom_xml = "<?xml version='1.0' encoding='utf-8'?>\n<data>\n  <row>\n    <index>0</index>\n    <shape>square</shape>\n    <degrees>360</degrees>\n    <sides>4.0</sides>\n  </row>\n  <row>\n    <index>1</index>\n    <shape>circle</shape>\n    <degrees>360</degrees>\n    <sides/>\n  </row>\n  <row>\n    <index>2</index>\n    <shape>triangle</shape>\n    <degrees>180</degrees>\n    <sides>3.0</sides>\n  </row>\n</data>"

def test_compression_output(parser, compression_only, geom_df):
    if False:
        while True:
            i = 10
    with tm.ensure_clean() as path:
        geom_df.to_xml(path, parser=parser, compression=compression_only)
        with get_handle(path, 'r', compression=compression_only) as handle_obj:
            output = handle_obj.handle.read()
    output = equalize_decl(output)
    assert geom_xml == output.strip()

def test_filename_and_suffix_comp(parser, compression_only, geom_df, compression_to_extension):
    if False:
        i = 10
        return i + 15
    compfile = 'xml.' + compression_to_extension[compression_only]
    with tm.ensure_clean(filename=compfile) as path:
        geom_df.to_xml(path, parser=parser, compression=compression_only)
        with get_handle(path, 'r', compression=compression_only) as handle_obj:
            output = handle_obj.handle.read()
    output = equalize_decl(output)
    assert geom_xml == output.strip()

def test_ea_dtypes(any_numeric_ea_dtype, parser):
    if False:
        i = 10
        return i + 15
    expected = "<?xml version='1.0' encoding='utf-8'?>\n<data>\n  <row>\n    <index>0</index>\n    <a/>\n  </row>\n</data>"
    df = DataFrame({'a': [NA]}).astype(any_numeric_ea_dtype)
    result = df.to_xml(parser=parser)
    assert equalize_decl(result).strip() == expected

def test_unsuported_compression(parser, geom_df):
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError, match='Unrecognized compression type'):
        with tm.ensure_clean() as path:
            geom_df.to_xml(path, parser=parser, compression='7z')

@pytest.mark.single_cpu
def test_s3_permission_output(parser, s3_public_bucket, geom_df):
    if False:
        print('Hello World!')
    s3fs = pytest.importorskip('s3fs')
    pytest.importorskip('lxml')
    with tm.external_error_raised((PermissionError, FileNotFoundError)):
        fs = s3fs.S3FileSystem(anon=True)
        fs.ls(s3_public_bucket.name)
        geom_df.to_xml(f's3://{s3_public_bucket.name}/geom.xml', compression='zip', parser=parser)