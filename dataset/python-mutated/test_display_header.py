from __future__ import annotations
from ansible.cli.galaxy import _display_header

def test_display_header_default(capsys):
    if False:
        print('Hello World!')
    _display_header('/collections/path', 'h1', 'h2')
    (out, err) = capsys.readouterr()
    out_lines = out.splitlines()
    assert out_lines[0] == ''
    assert out_lines[1] == '# /collections/path'
    assert out_lines[2] == 'h1         h2     '
    assert out_lines[3] == '---------- -------'

def test_display_header_widths(capsys):
    if False:
        for i in range(10):
            print('nop')
    _display_header('/collections/path', 'Collection', 'Version', 18, 18)
    (out, err) = capsys.readouterr()
    out_lines = out.splitlines()
    assert out_lines[0] == ''
    assert out_lines[1] == '# /collections/path'
    assert out_lines[2] == 'Collection         Version           '
    assert out_lines[3] == '------------------ ------------------'

def test_display_header_small_widths(capsys):
    if False:
        while True:
            i = 10
    _display_header('/collections/path', 'Col', 'Ver', 1, 1)
    (out, err) = capsys.readouterr()
    out_lines = out.splitlines()
    assert out_lines[0] == ''
    assert out_lines[1] == '# /collections/path'
    assert out_lines[2] == 'Col Ver'
    assert out_lines[3] == '--- ---'