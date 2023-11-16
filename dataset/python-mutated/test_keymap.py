from unittest import mock
import pytest
from mitmproxy.test import taddons
from mitmproxy.tools.console import keymap

def test_binding():
    if False:
        while True:
            i = 10
    b = keymap.Binding('space', 'cmd', ['options'], '')
    assert b.keyspec() == ' '

def test_bind():
    if False:
        while True:
            i = 10
    with taddons.context() as tctx:
        km = keymap.Keymap(tctx.master)
        km.executor = mock.Mock()
        with pytest.raises(ValueError):
            km.add('foo', 'bar', ['unsupported'])
        km.add('key', 'str', ['options', 'commands'])
        assert km.get('options', 'key')
        assert km.get('commands', 'key')
        assert not km.get('flowlist', 'key')
        assert len(km.list('commands')) == 1
        km.handle('unknown', 'unknown')
        assert not km.executor.called
        km.handle('options', 'key')
        assert km.executor.called
        km.add('glob', 'str', ['global'])
        km.executor = mock.Mock()
        km.handle('options', 'glob')
        assert km.executor.called
        assert len(km.list('global')) == 1

def test_join():
    if False:
        return 10
    with taddons.context() as tctx:
        km = keymap.Keymap(tctx.master)
        km.add('key', 'str', ['options'], 'help1')
        km.add('key', 'str', ['commands'])
        assert len(km.bindings) == 1
        assert len(km.bindings[0].contexts) == 2
        assert km.bindings[0].help == 'help1'
        km.add('key', 'str', ['commands'], 'help2')
        assert len(km.bindings) == 1
        assert len(km.bindings[0].contexts) == 2
        assert km.bindings[0].help == 'help2'
        assert km.get('commands', 'key')
        km.unbind(km.bindings[0])
        assert len(km.bindings) == 0
        assert not km.get('commands', 'key')

def test_remove():
    if False:
        return 10
    with taddons.context() as tctx:
        km = keymap.Keymap(tctx.master)
        km.add('key', 'str', ['options', 'commands'], 'help1')
        assert len(km.bindings) == 1
        assert 'options' in km.bindings[0].contexts
        km.remove('key', ['options'])
        assert len(km.bindings) == 1
        assert 'options' not in km.bindings[0].contexts
        km.remove('key', ['commands'])
        assert len(km.bindings) == 0

def test_load_path(tmpdir):
    if False:
        print('Hello World!')
    dst = str(tmpdir.join('conf'))
    with taddons.context() as tctx:
        kmc = keymap.KeymapConfig(tctx.master)
        km = keymap.Keymap(tctx.master)
        tctx.master.keymap = km
        with open(dst, 'wb') as f:
            f.write(b'\xff\xff\xff')
        with pytest.raises(keymap.KeyBindingError, match='expected UTF8'):
            kmc.load_path(km, dst)
        with open(dst, 'w') as f:
            f.write("'''")
        with pytest.raises(keymap.KeyBindingError):
            kmc.load_path(km, dst)
        with open(dst, 'w') as f:
            f.write('\n                    -   key: key1\n                        ctx: [unknown]\n                        cmd: >\n                            foo bar\n                            foo bar\n                ')
        with pytest.raises(keymap.KeyBindingError):
            kmc.load_path(km, dst)
        with open(dst, 'w') as f:
            f.write('\n                    -   key: key1\n                        ctx: [chooser]\n                        help: one\n                        cmd: >\n                            foo bar\n                            foo bar\n                ')
        kmc.load_path(km, dst)
        assert km.get('chooser', 'key1')
        with open(dst, 'w') as f:
            f.write('\n                    -   key: key2\n                        ctx: [flowlist]\n                        cmd: foo\n                    -   key: key2\n                        ctx: [flowview]\n                        cmd: bar\n                ')
        kmc.load_path(km, dst)
        assert km.get('flowlist', 'key2')
        assert km.get('flowview', 'key2')
        km.add('key123', 'str', ['flowlist', 'flowview'])
        with open(dst, 'w') as f:
            f.write('\n                    -   key: key123\n                        ctx: [options]\n                        cmd: foo\n                ')
        kmc.load_path(km, dst)
        assert km.get('flowlist', 'key123')
        assert km.get('flowview', 'key123')
        assert km.get('options', 'key123')

def test_parse():
    if False:
        print('Hello World!')
    with taddons.context() as tctx:
        kmc = keymap.KeymapConfig(tctx.master)
        assert kmc.parse('') == []
        assert kmc.parse('\n\n\n   \n') == []
        with pytest.raises(keymap.KeyBindingError, match='expected a list of keys'):
            kmc.parse('key: val')
        with pytest.raises(keymap.KeyBindingError, match='expected a list of keys'):
            kmc.parse('val')
        with pytest.raises(keymap.KeyBindingError, match='Unknown key attributes'):
            kmc.parse('\n                    -   key: key1\n                        nonexistent: bar\n                ')
        with pytest.raises(keymap.KeyBindingError, match='Missing required key attributes'):
            kmc.parse('\n                    -   help: key1\n                ')
        with pytest.raises(keymap.KeyBindingError, match='Invalid type for cmd'):
            kmc.parse('\n                    -   key: key1\n                        cmd: [ cmd ]\n                ')
        with pytest.raises(keymap.KeyBindingError, match='Invalid type for ctx'):
            kmc.parse('\n                    -   key: key1\n                        ctx: foo\n                        cmd: cmd\n                ')
        assert kmc.parse('\n                -   key: key1\n                    ctx: [one, two]\n                    help: one\n                    cmd: >\n                        foo bar\n                        foo bar\n            ') == [{'key': 'key1', 'ctx': ['one', 'two'], 'help': 'one', 'cmd': 'foo bar foo bar\n'}]