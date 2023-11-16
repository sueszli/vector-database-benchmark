"""Test the build process with Text builder with the test root."""
import pytest
from docutils.utils import column_width
from sphinx.writers.text import MAXWIDTH, Cell, Table

def with_text_app(*args, **kw):
    if False:
        for i in range(10):
            print('nop')
    default_kw = {'buildername': 'text', 'testroot': 'build-text'}
    default_kw.update(kw)
    return pytest.mark.sphinx(*args, **default_kw)

@with_text_app()
def test_maxwitdh_with_prefix(app, status, warning):
    if False:
        print('Hello World!')
    app.builder.build_update()
    result = (app.outdir / 'maxwidth.txt').read_text(encoding='utf8')
    lines = result.splitlines()
    line_widths = [column_width(line) for line in lines]
    assert max(line_widths) < MAXWIDTH
    assert lines[0].startswith('See also:')
    assert lines[1].startswith('')
    assert lines[2].startswith('  ham')
    assert lines[3].startswith('  ham')
    assert lines[4] == ''
    assert lines[5].startswith('* ham')
    assert lines[6].startswith('  ham')
    assert lines[7] == ''
    assert lines[8].startswith('* ham')
    assert lines[9].startswith('  ham')
    assert lines[10] == ''
    assert lines[11].startswith('spam egg')

@with_text_app()
def test_lineblock(app, status, warning):
    if False:
        print('Hello World!')
    app.builder.build_update()
    result = (app.outdir / 'lineblock.txt').read_text(encoding='utf8')
    expect = '* one\n\n     line-block 1\n     line-block 2\n\nfollowed paragraph.\n'
    assert result == expect

@with_text_app()
def test_nonascii_title_line(app, status, warning):
    if False:
        while True:
            i = 10
    app.builder.build_update()
    result = (app.outdir / 'nonascii_title.txt').read_text(encoding='utf8')
    expect_underline = '*********'
    result_underline = result.splitlines()[1].strip()
    assert expect_underline == result_underline

@with_text_app()
def test_nonascii_table(app, status, warning):
    if False:
        i = 10
        return i + 15
    app.builder.build_update()
    result = (app.outdir / 'nonascii_table.txt').read_text(encoding='utf8')
    lines = [line.strip() for line in result.splitlines() if line.strip()]
    line_widths = [column_width(line) for line in lines]
    assert len(set(line_widths)) == 1

@with_text_app()
def test_nonascii_maxwidth(app, status, warning):
    if False:
        print('Hello World!')
    app.builder.build_update()
    result = (app.outdir / 'nonascii_maxwidth.txt').read_text(encoding='utf8')
    lines = [line.strip() for line in result.splitlines() if line.strip()]
    line_widths = [column_width(line) for line in lines]
    assert max(line_widths) < MAXWIDTH

def test_table_builder():
    if False:
        for i in range(10):
            print('nop')
    table = Table([6, 6])
    table.add_cell(Cell('foo'))
    table.add_cell(Cell('bar'))
    table_str = str(table).split('\n')
    assert table_str[0] == '+--------+--------+'
    assert table_str[1] == '| foo    | bar    |'
    assert table_str[2] == '+--------+--------+'
    assert repr(table).count('<Cell ') == 2

def test_table_separator():
    if False:
        for i in range(10):
            print('nop')
    table = Table([6, 6])
    table.add_cell(Cell('foo'))
    table.add_cell(Cell('bar'))
    table.set_separator()
    table.add_row()
    table.add_cell(Cell('FOO'))
    table.add_cell(Cell('BAR'))
    table_str = str(table).split('\n')
    assert table_str[0] == '+--------+--------+'
    assert table_str[1] == '| foo    | bar    |'
    assert table_str[2] == '|========|========|'
    assert table_str[3] == '| FOO    | BAR    |'
    assert table_str[4] == '+--------+--------+'
    assert repr(table).count('<Cell ') == 4

def test_table_cell():
    if False:
        for i in range(10):
            print('nop')
    cell = Cell('Foo bar baz')
    cell.wrap(3)
    assert 'Cell' in repr(cell)
    assert cell.wrapped == ['Foo', 'bar', 'baz']

@with_text_app()
def test_table_with_empty_cell(app, status, warning):
    if False:
        return 10
    app.builder.build_update()
    result = (app.outdir / 'table.txt').read_text(encoding='utf8')
    lines = [line.strip() for line in result.splitlines() if line.strip()]
    assert lines[0] == '+-------+-------+'
    assert lines[1] == '| XXX   | XXX   |'
    assert lines[2] == '+-------+-------+'
    assert lines[3] == '|       | XXX   |'
    assert lines[4] == '+-------+-------+'
    assert lines[5] == '| XXX   |       |'
    assert lines[6] == '+-------+-------+'

@with_text_app()
def test_table_with_rowspan(app, status, warning):
    if False:
        while True:
            i = 10
    app.builder.build_update()
    result = (app.outdir / 'table_rowspan.txt').read_text(encoding='utf8')
    lines = [line.strip() for line in result.splitlines() if line.strip()]
    assert lines[0] == '+-------+-------+'
    assert lines[1] == '| XXXXXXXXX     |'
    assert lines[2] == '+-------+-------+'
    assert lines[3] == '|       | XXX   |'
    assert lines[4] == '+-------+-------+'
    assert lines[5] == '| XXX   |       |'
    assert lines[6] == '+-------+-------+'

@with_text_app()
def test_table_with_colspan(app, status, warning):
    if False:
        print('Hello World!')
    app.builder.build_update()
    result = (app.outdir / 'table_colspan.txt').read_text(encoding='utf8')
    lines = [line.strip() for line in result.splitlines() if line.strip()]
    assert lines[0] == '+-------+-------+'
    assert lines[1] == '| XXX   | XXX   |'
    assert lines[2] == '+-------+-------+'
    assert lines[3] == '|       | XXX   |'
    assert lines[4] == '+-------+       |'
    assert lines[5] == '| XXX   |       |'
    assert lines[6] == '+-------+-------+'

@with_text_app()
def test_table_with_colspan_left(app, status, warning):
    if False:
        i = 10
        return i + 15
    app.builder.build_update()
    result = (app.outdir / 'table_colspan_left.txt').read_text(encoding='utf8')
    lines = [line.strip() for line in result.splitlines() if line.strip()]
    assert lines[0] == '+-------+-------+'
    assert lines[1] == '| XXX   | XXX   |'
    assert lines[2] == '+-------+-------+'
    assert lines[3] == '| XXX   | XXX   |'
    assert lines[4] == '|       +-------+'
    assert lines[5] == '|       |       |'
    assert lines[6] == '+-------+-------+'

@with_text_app()
def test_table_with_colspan_and_rowspan(app, status, warning):
    if False:
        return 10
    app.builder.build_update()
    result = (app.outdir / 'table_colspan_and_rowspan.txt').read_text(encoding='utf8')
    lines = [line.strip() for line in result.splitlines() if line.strip()]
    assert result
    assert lines[0] == '+-------+-------+-------+'
    assert lines[1] == '| AAA           | BBB   |'
    assert lines[2] == '+-------+-------+       |'
    assert lines[3] == '| DDD   | XXX   |       |'
    assert lines[4] == '|       +-------+-------+'
    assert lines[5] == '|       | CCC           |'
    assert lines[6] == '+-------+-------+-------+'

@with_text_app()
def test_list_items_in_admonition(app, status, warning):
    if False:
        for i in range(10):
            print('nop')
    app.builder.build_update()
    result = (app.outdir / 'listitems.txt').read_text(encoding='utf8')
    lines = [line.rstrip() for line in result.splitlines()]
    assert lines[0] == 'See also:'
    assert lines[1] == ''
    assert lines[2] == '  * item 1'
    assert lines[3] == ''
    assert lines[4] == '  * item 2'

@with_text_app()
def test_secnums(app, status, warning):
    if False:
        return 10
    app.builder.build_all()
    index = (app.outdir / 'index.txt').read_text(encoding='utf8')
    lines = index.splitlines()
    assert lines[0] == '* 1. Section A'
    assert lines[1] == ''
    assert lines[2] == '* 2. Section B'
    assert lines[3] == ''
    assert lines[4] == '  * 2.1. Sub Ba'
    assert lines[5] == ''
    assert lines[6] == '  * 2.2. Sub Bb'
    doc2 = (app.outdir / 'doc2.txt').read_text(encoding='utf8')
    expect = '2. Section B\n************\n\n\n2.1. Sub Ba\n===========\n\n\n2.2. Sub Bb\n===========\n'
    assert doc2 == expect
    app.config.text_secnumber_suffix = ' '
    app.builder.build_all()
    index = (app.outdir / 'index.txt').read_text(encoding='utf8')
    lines = index.splitlines()
    assert lines[0] == '* 1 Section A'
    assert lines[1] == ''
    assert lines[2] == '* 2 Section B'
    assert lines[3] == ''
    assert lines[4] == '  * 2.1 Sub Ba'
    assert lines[5] == ''
    assert lines[6] == '  * 2.2 Sub Bb'
    doc2 = (app.outdir / 'doc2.txt').read_text(encoding='utf8')
    expect = '2 Section B\n***********\n\n\n2.1 Sub Ba\n==========\n\n\n2.2 Sub Bb\n==========\n'
    assert doc2 == expect
    app.config.text_add_secnumbers = False
    app.builder.build_all()
    index = (app.outdir / 'index.txt').read_text(encoding='utf8')
    lines = index.splitlines()
    assert lines[0] == '* Section A'
    assert lines[1] == ''
    assert lines[2] == '* Section B'
    assert lines[3] == ''
    assert lines[4] == '  * Sub Ba'
    assert lines[5] == ''
    assert lines[6] == '  * Sub Bb'
    doc2 = (app.outdir / 'doc2.txt').read_text(encoding='utf8')
    expect = 'Section B\n*********\n\n\nSub Ba\n======\n\n\nSub Bb\n======\n'
    assert doc2 == expect