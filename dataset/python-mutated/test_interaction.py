import os
import sys
import pexpect
import pytest
from .base import TempAppDirTestCase
from .utils import get_http_prompt_path
from http_prompt import config

class TestInteraction(TempAppDirTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(TestInteraction, self).setUp()
        self.orig_config_home = os.getenv('XDG_CONFIG_HOME')
        os.environ['XDG_CONFIG_HOME'] = self.temp_dir
        self.orig_term = os.getenv('TERM')
        os.environ['TERM'] = 'screen-256color'

    def tearDown(self):
        if False:
            return 10
        super(TestInteraction, self).tearDown()
        os.environ['XDG_CONFIG_HOME'] = self.orig_config_home
        if self.orig_term:
            os.environ['TERM'] = self.orig_term
        else:
            os.environ.pop('TERM', None)

    def write_config(self, content):
        if False:
            return 10
        config_path = config.get_user_config_path()
        with open(config_path, 'a') as f:
            f.write(content)

    @pytest.mark.skipif(sys.platform == 'win32', reason="pexpect doesn't work well on Windows")
    @pytest.mark.slow
    def test_interaction(self):
        if False:
            for i in range(10):
                print('nop')
        bin_path = get_http_prompt_path()
        child = pexpect.spawn(bin_path, env=os.environ)
        child.sendline('exit')
        child.expect_exact('Goodbye!', timeout=20)
        child.close()

    @pytest.mark.skipif(sys.platform == 'win32', reason="pexpect doesn't work well on Windows")
    @pytest.mark.slow
    def test_vi_mode(self):
        if False:
            i = 10
            return i + 15
        self.write_config('vi = True\n')
        bin_path = get_http_prompt_path()
        child = pexpect.spawn(bin_path, env=os.environ)
        child.expect_exact('http://localhost:8000>')
        child.send('htpie')
        child.send('\x1b')
        child.sendline('hhit')
        child.expect_exact('http http://localhost:8000')
        child.send('\x1b')
        child.send('i')
        child.sendline('exit')
        child.expect_exact('Goodbye!', timeout=20)
        child.close()