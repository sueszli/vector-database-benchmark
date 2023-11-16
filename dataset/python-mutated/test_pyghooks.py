"""Tests pygments hooks."""
import os
import pathlib
import stat
from tempfile import TemporaryDirectory
import pytest
from xonsh.environ import LsColors
from xonsh.platform import ON_WINDOWS
from xonsh.pyghooks import XSH, Color, Token, XonshLexer, XonshStyle, code_by_name, color_file, color_name_to_pygments_code, file_color_tokens, get_style_by_name, register_custom_pygments_style

@pytest.fixture
def xs_LS_COLORS(xession, os_env, monkeypatch):
    if False:
        i = 10
        return i + 15
    'Xonsh environment including LS_COLORS'
    monkeypatch.setattr(xession, 'env', os_env)
    lsc = LsColors(LsColors.default_settings)
    xession.env['LS_COLORS'] = lsc
    xession.env['INTENSIFY_COLORS_ON_WIN'] = False
    xession.shell.shell_type = 'prompt_toolkit'
    xession.shell.shell.styler = XonshStyle()
    yield xession
DEFAULT_STYLES = {Color.RESET: 'noinherit', Color.BLACK: 'ansiblack', Color.BLUE: 'ansiblue', Color.CYAN: 'ansicyan', Color.GREEN: 'ansigreen', Color.PURPLE: 'ansimagenta', Color.RED: 'ansired', Color.WHITE: 'ansigray', Color.YELLOW: 'ansiyellow', Color.INTENSE_BLACK: 'ansibrightblack', Color.INTENSE_BLUE: 'ansibrightblue', Color.INTENSE_CYAN: 'ansibrightcyan', Color.INTENSE_GREEN: 'ansibrightgreen', Color.INTENSE_PURPLE: 'ansibrightmagenta', Color.INTENSE_RED: 'ansibrightred', Color.INTENSE_WHITE: 'ansiwhite', Color.INTENSE_YELLOW: 'ansibrightyellow'}

@pytest.mark.parametrize('name, exp', [('RESET', 'noinherit'), ('RED', 'ansired'), ('BACKGROUND_RED', 'bg:ansired'), ('BACKGROUND_INTENSE_RED', 'bg:ansibrightred'), ('BOLD_RED', 'bold ansired'), ('UNDERLINE_RED', 'underline ansired'), ('BOLD_UNDERLINE_RED', 'bold underline ansired'), ('UNDERLINE_BOLD_RED', 'underline bold ansired'), ('BOLD_FAINT_RED', 'bold ansired'), ('BOLD_SLOWBLINK_RED', 'bold ansired'), ('BOLD_FASTBLINK_RED', 'bold ansired'), ('BOLD_INVERT_RED', 'bold ansired'), ('BOLD_CONCEAL_RED', 'bold ansired'), ('BOLD_STRIKETHROUGH_RED', 'bold ansired'), ('#000', '#000'), ('#000000', '#000000'), ('BACKGROUND_#000', 'bg:#000'), ('BACKGROUND_#000000', 'bg:#000000'), ('BG#000', 'bg:#000'), ('bg#000000', 'bg:#000000')])
def test_color_name_to_pygments_code(name, exp):
    if False:
        while True:
            i = 10
    styles = DEFAULT_STYLES.copy()
    obs = color_name_to_pygments_code(name, styles)
    assert obs == exp

@pytest.mark.parametrize('name, exp', [('RESET', 'noinherit'), ('RED', 'ansired'), ('BACKGROUND_RED', 'bg:ansired'), ('BACKGROUND_INTENSE_RED', 'bg:ansibrightred'), ('BOLD_RED', 'bold ansired'), ('UNDERLINE_RED', 'underline ansired'), ('BOLD_UNDERLINE_RED', 'bold underline ansired'), ('UNDERLINE_BOLD_RED', 'underline bold ansired'), ('BOLD_FAINT_RED', 'bold ansired'), ('BOLD_SLOWBLINK_RED', 'bold ansired'), ('BOLD_FASTBLINK_RED', 'bold ansired'), ('BOLD_INVERT_RED', 'bold ansired'), ('BOLD_CONCEAL_RED', 'bold ansired'), ('BOLD_STRIKETHROUGH_RED', 'bold ansired'), ('#000', '#000'), ('#000000', '#000000'), ('BACKGROUND_#000', 'bg:#000'), ('BACKGROUND_#000000', 'bg:#000000'), ('BG#000', 'bg:#000'), ('bg#000000', 'bg:#000000')])
def test_code_by_name(name, exp):
    if False:
        print('Hello World!')
    styles = DEFAULT_STYLES.copy()
    obs = code_by_name(name, styles)
    assert obs == exp

@pytest.mark.parametrize('in_tuple, exp_ct, exp_ansi_colors', [(('RESET',), Color.RESET, 'noinherit'), (('GREEN',), Color.GREEN, 'ansigreen'), (('BOLD_RED',), Color.BOLD_RED, 'bold ansired'), (('BACKGROUND_BLACK', 'BOLD_GREEN'), Color.BACKGROUND_BLACK__BOLD_GREEN, 'bg:ansiblack bold ansigreen')])
def test_color_token_by_name(in_tuple, exp_ct, exp_ansi_colors, xs_LS_COLORS):
    if False:
        for i in range(10):
            print('nop')
    from xonsh.pyghooks import XonshStyle, color_token_by_name
    xs = XonshStyle()
    styles = xs.styles
    ct = color_token_by_name(in_tuple, styles)
    ansi_colors = styles[ct]
    assert ct == exp_ct, 'returned color token is right'
    assert ansi_colors == exp_ansi_colors, 'color token mapped to correct color string'

def test_XonshStyle_init_file_color_tokens(xs_LS_COLORS, monkeypatch):
    if False:
        i = 10
        return i + 15
    keys = list(file_color_tokens)
    for n in keys:
        monkeypatch.delitem(file_color_tokens, n)
    xs = XonshStyle()
    assert xs.styles
    assert isinstance(file_color_tokens, dict)
    assert set(file_color_tokens.keys()) == set(xs_LS_COLORS.env['LS_COLORS'].keys())
if ON_WINDOWS:
    _cf = {'fi': 'regular', 'di': 'simple_dir', 'ln': 'sym_link', 'pi': None, 'so': None, 'do': None, 'or': 'orphan', 'mi': None, 'su': None, 'sg': None, 'ca': None, 'tw': None, 'ow': None, 'st': None, 'ex': None, '*.emf': 'foo.emf', '*.zip': 'foo.zip', '*.ogg': 'foo.ogg', 'mh': 'hard_link'}
else:
    _cf = {'fi': 'regular', 'di': 'simple_dir', 'ln': 'sym_link', 'pi': 'pipe', 'so': None, 'do': None, 'or': 'orphan', 'mi': None, 'su': 'set_uid', 'sg': 'set_gid', 'ca': None, 'tw': 'sticky_ow_dir', 'ow': 'other_writable_dir', 'st': 'sticky_dir', 'ex': 'executable', '*.emf': 'foo.emf', '*.zip': 'foo.zip', '*.ogg': 'foo.ogg', 'mh': 'hard_link'}

@pytest.fixture(scope='module')
def colorizable_files():
    if False:
        while True:
            i = 10
    'populate temp dir with sample files.\n    (too hard to emit indivual test cases when fixture invoked in mark.parametrize)'
    with TemporaryDirectory() as tempdir:
        for (k, v) in _cf.items():
            if v is None:
                continue
            if v.startswith('/'):
                file_path = v
            else:
                file_path = tempdir + '/' + v
            try:
                os.lstat(file_path)
            except FileNotFoundError:
                if file_path.endswith('_dir'):
                    os.mkdir(file_path)
                else:
                    open(file_path, 'a').close()
                if k in ('di', 'fi'):
                    pass
                elif k == 'ex':
                    os.chmod(file_path, stat.S_IRWXU)
                elif k == 'ln':
                    os.chmod(file_path, stat.S_IRWXU)
                    os.rename(file_path, file_path + '_target')
                    os.symlink(file_path + '_target', file_path)
                elif k == 'or':
                    os.rename(file_path, file_path + '_target')
                    os.symlink(file_path + '_target', file_path)
                    os.remove(file_path + '_target')
                elif k == 'pi':
                    os.remove(file_path)
                    os.mkfifo(file_path)
                elif k == 'su':
                    os.chmod(file_path, stat.S_ISUID)
                elif k == 'sg':
                    os.chmod(file_path, stat.S_ISGID)
                elif k == 'st':
                    os.chmod(file_path, stat.S_ISVTX | stat.S_IRUSR | stat.S_IWUSR)
                elif k == 'tw':
                    os.chmod(file_path, stat.S_ISVTX | stat.S_IWOTH | stat.S_IRUSR | stat.S_IWUSR)
                elif k == 'ow':
                    os.chmod(file_path, stat.S_IWOTH | stat.S_IRUSR | stat.S_IWUSR)
                elif k == 'mh':
                    os.rename(file_path, file_path + '_target')
                    os.link(file_path + '_target', file_path)
                else:
                    pass
                os.symlink(file_path, file_path + '_symlink')
        yield tempdir
    pass

@pytest.mark.parametrize('key,file_path', [(key, file_path) for (key, file_path) in _cf.items() if file_path])
def test_colorize_file(key, file_path, colorizable_files, xs_LS_COLORS):
    if False:
        return 10
    'test proper file codes with symlinks colored normally'
    ffp = colorizable_files + '/' + file_path
    stat_result = os.lstat(ffp)
    (color_token, color_key) = color_file(ffp, stat_result)
    assert color_key == key, 'File classified as expected kind'
    assert color_token == file_color_tokens[key], 'Color token is as expected'

@pytest.mark.parametrize('key,file_path', [(key, file_path) for (key, file_path) in _cf.items() if file_path])
def test_colorize_file_symlink(key, file_path, colorizable_files, xs_LS_COLORS):
    if False:
        return 10
    'test proper file codes with symlinks colored target.'
    xs_LS_COLORS.env['LS_COLORS']['ln'] = 'target'
    ffp = colorizable_files + '/' + file_path + '_symlink'
    stat_result = os.lstat(ffp)
    assert stat.S_ISLNK(stat_result.st_mode)
    (_, color_key) = color_file(ffp, stat_result)
    try:
        tar_stat_result = os.stat(ffp)
        tar_ffp = str(pathlib.Path(ffp).resolve())
        (_, tar_color_key) = color_file(tar_ffp, tar_stat_result)
        if tar_color_key.startswith('*'):
            tar_color_key = 'fi'
    except FileNotFoundError:
        tar_color_key = 'or'
    assert color_key == tar_color_key, 'File classified as expected kind, via symlink'
import xonsh.lazyimps

def test_colorize_file_ca(xs_LS_COLORS, monkeypatch):
    if False:
        i = 10
        return i + 15

    def mock_os_listxattr(*args, **kwards):
        if False:
            for i in range(10):
                print('nop')
        return ['security.capability']
    monkeypatch.setattr(xonsh.pyghooks, 'os_listxattr', mock_os_listxattr)
    with TemporaryDirectory() as tmpdir:
        file_path = tmpdir + '/cap_file'
        open(file_path, 'a').close()
        os.chmod(file_path, stat.S_IRWXU)
        (color_token, color_key) = color_file(file_path, os.lstat(file_path))
        assert color_key == 'ca'

@pytest.mark.parametrize('name, styles, refrules', [('test1', {}, {}), ('test2', {Token.Literal.String.Single: '#ff0000'}, {Token.Literal.String.Single: '#ff0000'}), ('test3', {'Token.Literal.String.Single': '#ff0000'}, {Token.Literal.String.Single: '#ff0000'}), ('test4', {'Literal.String.Single': '#ff0000'}, {Token.Literal.String.Single: '#ff0000'}), ('test5', {'completion-menu.completion.current': '#00ff00'}, {Token.PTK.CompletionMenu.Completion.Current: '#00ff00'}), ('test6', {'RED': '#ff0000'}, {Token.Color.RED: '#ff0000'})])
def test_register_custom_pygments_style(name, styles, refrules):
    if False:
        return 10
    register_custom_pygments_style(name, styles)
    style = get_style_by_name(name)
    assert style is not None
    for (rule, color) in refrules.items():
        assert rule in style.styles
        assert style.styles[rule] == color

def test_can_use_xonsh_lexer_without_xession(xession, monkeypatch):
    if False:
        while True:
            i = 10
    monkeypatch.setattr(xession, 'env', None)
    assert XSH.env is None
    lexer = XonshLexer()
    assert XSH.env is not None
    list(lexer.get_tokens_unprocessed('  some text'))