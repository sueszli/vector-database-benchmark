"""Dynamic configuration files tests."""
from __future__ import unicode_literals, division, absolute_import, print_function
import sys
import os
import json
import logging
import tests.modules.vim as vim_module
from tests.modules.lib import Args, urllib_read, replace_attr
from tests.modules import TestCase
from powerline import NotInterceptedError
from powerline.segments.common import wthr
VBLOCK = chr(ord('V') - 64)
SBLOCK = chr(ord('S') - 64)

class FailingLogger(logging.Logger):

    def exception(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(FailingLogger, self).exception(*args, **kwargs)
        raise NotInterceptedError('Unexpected exception occurred')

def get_logger(stream=None):
    if False:
        for i in range(10):
            print('nop')
    log_format = '%(asctime)s:%(levelname)s:%(message)s'
    formatter = logging.Formatter(log_format)
    level = logging.WARNING
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger = FailingLogger('powerline')
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

class TestVimConfig(TestCase):

    def test_vim(self):
        if False:
            i = 10
            return i + 15
        from powerline.vim import VimPowerline
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'powerline', 'config_files')
        buffers = ((('bufoptions',), {'buftype': 'help'}), (('bufname', '[Command Line]'), {}), (('bufoptions',), {'buftype': 'quickfix'}), (('bufname', 'NERD_tree_1'), {}), (('bufname', '__Gundo__'), {}), (('bufname', '__Gundo_Preview__'), {}))
        with open(os.path.join(cfg_path, 'config.json'), 'r') as f:
            local_themes_raw = json.load(f)['ext']['vim']['local_themes']
            local_themes = dict(((k, v) for (k, v) in local_themes_raw.items()))
            self.assertEqual(len(buffers), len(local_themes) - 2)
        outputs = {}
        i = 0
        with vim_module._with('split'):
            with VimPowerline(logger=get_logger()) as powerline:

                def check_output(mode, args, kwargs):
                    if False:
                        print('Hello World!')
                    if mode == 'nc':
                        window = vim_module.windows[0]
                        window_id = 2
                    else:
                        vim_module._start_mode(mode)
                        window = vim_module.current.window
                        window_id = 1
                    winnr = window.number
                    out = powerline.render(window, window_id, winnr)
                    if out in outputs:
                        self.fail('Duplicate in set #{0} ({1}) for mode {2!r} (previously defined in set #{3} ({4!r}) for mode {5!r})'.format(i, (args, kwargs), mode, *outputs[out]))
                    outputs[out] = (i, (args, kwargs), mode)
                with vim_module._with('bufname', '/tmp/foo.txt'):
                    out = powerline.render(vim_module.current.window, 1, vim_module.current.window.number, is_tabline=True)
                    outputs[out] = (-1, (None, None), 'tab')
                    with vim_module._with('globals', powerline_config_paths=[cfg_path]):
                        exclude = set(('no', 'v', 'V', VBLOCK, 's', 'S', SBLOCK, 'R', 'Rv', 'c', 'cv', 'ce', 'r', 'rm', 'r?', '!'))
                        try:
                            for mode in ['n', 'nc', 'no', 'v', 'V', VBLOCK, 's', 'S', SBLOCK, 'i', 'R', 'Rv', 'c', 'cv', 'ce', 'r', 'rm', 'r?', '!']:
                                check_output(mode, None, None)
                                for (args, kwargs) in buffers:
                                    i += 1
                                    if mode in exclude:
                                        continue
                                    with vim_module._with(*args, **kwargs):
                                        check_output(mode, args, kwargs)
                        finally:
                            vim_module._start_mode('n')

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vim_sys_path')))

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        sys.path.pop(0)

class TestConfig(TestCase):

    def test_tmux(self):
        if False:
            print('Hello World!')
        from powerline.segments import common
        from imp import reload
        reload(common)
        from powerline.shell import ShellPowerline
        with replace_attr(common, 'urllib_read', urllib_read):
            with ShellPowerline(Args(ext=['tmux']), logger=get_logger(), run_once=False) as powerline:
                powerline.render()
            with ShellPowerline(Args(ext=['tmux']), logger=get_logger(), run_once=False) as powerline:
                powerline.render()

    def test_zsh(self):
        if False:
            print('Hello World!')
        from powerline.shell import ShellPowerline
        args = Args(last_pipe_status=[1, 0], jobnum=0, ext=['shell'], renderer_module='.zsh')
        segment_info = {'args': args}
        with ShellPowerline(args, logger=get_logger(), run_once=False) as powerline:
            powerline.render(segment_info=segment_info)
        with ShellPowerline(args, logger=get_logger(), run_once=False) as powerline:
            powerline.render(segment_info=segment_info)
        segment_info['local_theme'] = 'select'
        with ShellPowerline(args, logger=get_logger(), run_once=False) as powerline:
            powerline.render(segment_info=segment_info)
        segment_info['local_theme'] = 'continuation'
        segment_info['parser_state'] = 'if cmdsubst'
        with ShellPowerline(args, logger=get_logger(), run_once=False) as powerline:
            powerline.render(segment_info=segment_info)

    def test_bash(self):
        if False:
            while True:
                i = 10
        from powerline.shell import ShellPowerline
        args = Args(last_exit_code=1, last_pipe_status=[], jobnum=0, ext=['shell'], renderer_module='.bash', config_override={'ext': {'shell': {'theme': 'default_leftonly'}}})
        with ShellPowerline(args, logger=get_logger(), run_once=False) as powerline:
            powerline.render(segment_info={'args': args})
        with ShellPowerline(args, logger=get_logger(), run_once=False) as powerline:
            powerline.render(segment_info={'args': args})

    def test_ipython(self):
        if False:
            return 10
        from powerline.ipython import IPythonPowerline

        class IpyPowerline(IPythonPowerline):
            config_paths = None
            config_overrides = None
            theme_overrides = {}
        segment_info = Args(prompt_count=1)
        with IpyPowerline(logger=get_logger(), renderer_module='.pre_5') as powerline:
            for prompt_type in ['in', 'in2']:
                powerline.render(is_prompt=True, matcher_info=prompt_type, segment_info=segment_info)
                powerline.render(is_prompt=True, matcher_info=prompt_type, segment_info=segment_info)
        with IpyPowerline(logger=get_logger(), renderer_module='.pre_5') as powerline:
            for prompt_type in ['out', 'rewrite']:
                powerline.render(is_prompt=False, matcher_info=prompt_type, segment_info=segment_info)
                powerline.render(is_prompt=False, matcher_info=prompt_type, segment_info=segment_info)

    def test_wm(self):
        if False:
            for i in range(10):
                print('nop')
        from powerline.segments import common
        from imp import reload
        reload(common)
        from powerline import Powerline
        with replace_attr(wthr, 'urllib_read', urllib_read):
            Powerline(logger=get_logger(), ext='wm', renderer_module='pango_markup', run_once=True).render()
        reload(common)
old_cwd = None
saved_get_config_paths = None

def setUpModule():
    if False:
        while True:
            i = 10
    global old_cwd
    global saved_get_config_paths
    import powerline
    saved_get_config_paths = powerline.get_config_paths
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'powerline', 'config_files')
    powerline.get_config_paths = lambda : [path]
    old_cwd = os.getcwd()

def tearDownModule():
    if False:
        while True:
            i = 10
    global old_cwd
    global saved_get_config_paths
    import powerline
    powerline.get_config_paths = saved_get_config_paths
    os.chdir(old_cwd)
    old_cwd = None
if __name__ == '__main__':
    from tests.modules import main
    main()