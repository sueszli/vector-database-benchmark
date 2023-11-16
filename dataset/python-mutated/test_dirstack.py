"""Testing dirstack"""
import os
import pytest
from xonsh import dirstack
from xonsh.tools import chdir
HERE = os.path.abspath(os.path.dirname(__file__))
PARENT = os.path.dirname(HERE)

def test_simple(xession):
    if False:
        return 10
    xession.env.update(dict(CDPATH=PARENT, PWD=PARENT))
    with chdir(PARENT):
        assert os.getcwd() != HERE
        dirstack.cd(['tests'])
        assert os.getcwd() == HERE

def test_cdpath_simple(xession):
    if False:
        for i in range(10):
            print('nop')
    xession.env.update(dict(CDPATH=PARENT, PWD=HERE))
    with chdir(os.path.normpath('/')):
        assert os.getcwd() != HERE
        dirstack.cd(['tests'])
        assert os.getcwd() == HERE

def test_cdpath_collision(xession):
    if False:
        return 10
    xession.env.update(dict(CDPATH=PARENT, PWD=HERE))
    sub_tests = os.path.join(HERE, 'tests')
    if not os.path.exists(sub_tests):
        os.mkdir(sub_tests)
    with chdir(HERE):
        assert os.getcwd() == HERE
        dirstack.cd(['tests'])
        assert os.getcwd() == os.path.join(HERE, 'tests')

def test_cdpath_expansion(xession):
    if False:
        for i in range(10):
            print('nop')
    xession.env.update(dict(HERE=HERE, CDPATH=('~', '$HERE')))
    test_dirs = (os.path.join(HERE, 'xonsh-test-cdpath-here'), os.path.expanduser('~/xonsh-test-cdpath-home'))
    try:
        for d in test_dirs:
            if not os.path.exists(d):
                os.mkdir(d)
            assert os.path.exists(dirstack._try_cdpath(d)), f'dirstack._try_cdpath: could not resolve {d}'
    finally:
        for d in test_dirs:
            if os.path.exists(d):
                os.rmdir(d)

def test_cdpath_events(xession, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    xession.env.update(dict(CDPATH=PARENT, PWD=os.getcwd()))
    target = str(tmpdir)
    ev = None

    @xession.builtins.events.on_chdir
    def handler(olddir, newdir, **kw):
        if False:
            i = 10
            return i + 15
        nonlocal ev
        ev = (olddir, newdir)
    old_dir = os.getcwd()
    try:
        dirstack.cd([target])
    except Exception:
        raise
    else:
        assert (old_dir, target) == ev
    finally:
        os.chdir(old_dir)

def test_cd_autopush(xession, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    xession.env.update(dict(CDPATH=PARENT, PWD=os.getcwd(), AUTO_PUSHD=True))
    target = str(tmpdir)
    old_dir = os.getcwd()
    old_ds_size = len(dirstack.DIRSTACK)
    assert target != old_dir
    try:
        dirstack.cd([target])
        assert target == os.getcwd()
        assert old_ds_size + 1 == len(dirstack.DIRSTACK)
        dirstack.popd([])
    except Exception:
        raise
    finally:
        while len(dirstack.DIRSTACK) > old_ds_size:
            dirstack.popd([])
    assert old_dir == os.getcwd()

def test_cd_home(xession, tmpdir):
    if False:
        print('Hello World!')
    target = str(tmpdir)
    old_home = xession.env.get('HOME')
    xession.env.update(dict(HOME=target, PWD=os.getcwd(), AUTO_PUSHD=True))
    dirstack.cd([])
    assert target == os.getcwd()
    dirstack.popd([])
    xession.env.update(dict(HOME=old_home))