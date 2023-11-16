from libqtile.extension.base import _Extension
from libqtile.extension.dmenu import Dmenu, DmenuRun, J4DmenuDesktop
BLACK = '#000000'

def test_dmenu_configuration_options():
    if False:
        while True:
            i = 10
    '\n    Test that configuration options are correctly translated into\n    command options for dmenu.\n    '
    _Extension.global_defaults = {}
    opts = [({}, ['dmenu']), ({'dmenu_command': 'testdmenu --test-option'}, ['testdmenu', '--test-option']), ({'dmenu_command': ['testdmenu', '--test-option']}, ['testdmenu', '--test-option']), ({}, ['-fn', 'sans']), ({'dmenu_font': 'testfont'}, ['-fn', 'testfont']), ({'font': 'testfont'}, ['-fn', 'testfont']), ({'font': 'testfont', 'fontsize': 12}, ['-fn', 'testfont-12']), ({'dmenu_bottom': True}, ['-b']), ({'dmenu_ignorecase': True}, ['-i']), ({'dmenu_lines': 5}, ['-l', '5']), ({'dmenu_prompt': 'testprompt'}, ['-p', 'testprompt']), ({'background': BLACK}, ['-nb', BLACK]), ({'foreground': BLACK}, ['-nf', BLACK]), ({'selected_background': BLACK}, ['-sb', BLACK]), ({'selected_foreground': BLACK}, ['-sf', BLACK]), ({'dmenu_height': 100}, ['-h', '100'])]
    for (config, output) in opts:
        extension = Dmenu(**config)
        extension._configure(None)
        index = extension.configured_command.index(output[0])
        assert output == extension.configured_command[index:index + len(output)]

def test_dmenu_run(monkeypatch):
    if False:
        while True:
            i = 10

    def fake_popen(cmd, *args, **kwargs):
        if False:
            i = 10
            return i + 15

        class PopenObj:

            def communicate(self, value_in, *args):
                if False:
                    for i in range(10):
                        print('nop')
                return [value_in, None]
        return PopenObj()
    monkeypatch.setattr('libqtile.extension.base.Popen', fake_popen)
    extension = Dmenu(dmenu_lines=5)
    extension._configure(None)
    items = ['test1', 'test2']
    assert extension.run(items) == 'test1\ntest2\n'
    assert extension.configured_command[-2:] == ['-l', '2']

def test_dmenurun_extension():
    if False:
        for i in range(10):
            print('nop')
    extension = DmenuRun()
    assert extension.dmenu_command == 'dmenu_run'

def test_j4dmenu_configuration_options():
    if False:
        while True:
            i = 10
    '\n    Test that configuration options are correctly translated into\n    command options for dmenu.\n    '
    _Extension.global_defaults = {}
    opts = [({}, ['j4-dmenu-desktop', '--dmenu']), ({'font': 'testfont'}, ['dmenu -fn testfont']), ({'j4dmenu_use_xdg_de': True}, ['--use-xdg-de']), ({'j4dmenu_display_binary': True}, ['--display-binary']), ({'j4dmenu_generic': False}, ['--no-generic']), ({'j4dmenu_terminal': 'testterminal'}, ['--term', 'testterminal']), ({'j4dmenu_usage_log': 'testlog'}, ['--usage-log', 'testlog'])]
    for (config, output) in opts:
        extension = J4DmenuDesktop(**config)
        extension._configure(None)
        index = extension.configured_command.index(output[0])
        print(extension.configured_command)
        assert output == extension.configured_command[index:index + len(output)]