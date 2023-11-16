import os
import shutil
import textwrap
import pytest
import salt.utils.files
import salt.utils.platform
import salt.utils.stringutils
from tests.support.case import ModuleCase

@pytest.mark.windows_whitelisted
class PyDSLRendererIncludeTestCase(ModuleCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.directory_created = False
        if salt.utils.platform.is_windows():
            if not os.path.isdir('\\tmp'):
                os.mkdir('\\tmp')
                self.directory_created = True

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        if salt.utils.platform.is_windows():
            if self.directory_created:
                shutil.rmtree('\\tmp')

    @pytest.mark.destructive_test
    @pytest.mark.slow_test
    def test_rendering_includes(self):
        if False:
            while True:
                i = 10
        '\n        This test is currently hard-coded to /tmp to work-around a seeming\n        inability to load custom modules inside the pydsl renderers. This\n        is a FIXME.\n        '
        self.run_function('state.sls', ['pydsl.aaa'])
        expected = textwrap.dedent('            X1\n            X2\n            X3\n            Y1 extended\n            Y2 extended\n            Y3\n            hello red 1\n            hello green 2\n            hello blue 3\n            ')
        if salt.utils.platform.is_windows():
            expected = 'X1 \r\nX2 \r\nX3 \r\nY1 extended \r\nY2 extended \r\nY3 \r\nhello red 1 \r\nhello green 2 \r\nhello blue 3 \r\n'
        try:
            with salt.utils.files.fopen('/tmp/output', 'r') as f:
                ret = salt.utils.stringutils.to_unicode(f.read())
        finally:
            os.remove('/tmp/output')
        self.assertEqual(sorted(ret), sorted(expected))