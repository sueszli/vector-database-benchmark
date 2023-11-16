from dvc.commands.ls.ls_colors import LsColors

def colorize(ls_colors):
    if False:
        return 10

    def _colorize(f, spec=''):
        if False:
            i = 10
            return i + 15
        fs_path = {'path': f, 'isexec': 'e' in spec, 'isdir': 'd' in spec, 'isout': 'o' in spec}
        return ls_colors.format(fs_path)
    return _colorize

def test_ls_colors_out_file():
    if False:
        while True:
            i = 10
    ls_colors = LsColors(LsColors.default)
    assert colorize(ls_colors)('file', 'o') == 'file'

def test_ls_colors_out_dir():
    if False:
        return 10
    ls_colors = LsColors(LsColors.default)
    assert colorize(ls_colors)('dir', 'do') == '\x1b[01;34mdir\x1b[0m'

def test_ls_colors_out_exec():
    if False:
        i = 10
        return i + 15
    ls_colors = LsColors(LsColors.default)
    assert colorize(ls_colors)('script.sh', 'eo') == '\x1b[01;32mscript.sh\x1b[0m'

def test_ls_colors_out_ext():
    if False:
        i = 10
        return i + 15
    ls_colors = LsColors(LsColors.default + ':*.xml=01;33')
    assert colorize(ls_colors)('file.xml', 'o') == '\x1b[01;33mfile.xml\x1b[0m'

def test_ls_colors_file():
    if False:
        i = 10
        return i + 15
    ls_colors = LsColors(LsColors.default)
    assert colorize(ls_colors)('file') == 'file'

def test_ls_colors_dir():
    if False:
        for i in range(10):
            print('nop')
    ls_colors = LsColors(LsColors.default)
    assert colorize(ls_colors)('dir', 'd') == '\x1b[01;34mdir\x1b[0m'

def test_ls_colors_exec():
    if False:
        while True:
            i = 10
    ls_colors = LsColors(LsColors.default)
    assert colorize(ls_colors)('script.sh', 'e') == '\x1b[01;32mscript.sh\x1b[0m'

def test_ls_colors_ext():
    if False:
        for i in range(10):
            print('nop')
    ls_colors = LsColors(LsColors.default + ':*.xml=01;33')
    assert colorize(ls_colors)('file.xml') == '\x1b[01;33mfile.xml\x1b[0m'

def test_ls_repo_with_custom_color_env_defined(monkeypatch):
    if False:
        i = 10
        return i + 15
    monkeypatch.setenv('LS_COLORS', 'rs=0:di=01;34:*.xml=01;31:*.dvc=01;33:')
    ls_colors = LsColors()
    colorizer = colorize(ls_colors)
    assert colorizer('.dvcignore') == '.dvcignore'
    assert colorizer('.gitignore') == '.gitignore'
    assert colorizer('README.md') == 'README.md'
    assert colorizer('data', 'd') == '\x1b[01;34mdata\x1b[0m'
    assert colorizer('structure.xml') == '\x1b[01;31mstructure.xml\x1b[0m'
    assert colorizer('structure.xml.dvc') == '\x1b[01;33mstructure.xml.dvc\x1b[0m'