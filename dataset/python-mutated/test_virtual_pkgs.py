import os
import platform
import pytest
from .helpers import info

class TestVirtualPkgs:

    def test_virtual_packages(self):
        if False:
            return 10
        infos = info()
        assert 'virtual packages :' in infos
        assert '__archspec=1=' in infos
        if platform.system() == 'Windows':
            assert '__win' in infos
        elif platform.system() == 'Darwin':
            assert '__unix=0=0' in infos
            assert '__osx' in infos
        elif platform.system() == 'Linux':
            assert '__unix=0=0' in infos
            assert '__glibc' in infos
            linux_ver = platform.release().split('-', 1)[0]
            assert f'__linux={linux_ver}=0' in infos

    def test_virtual_linux(self):
        if False:
            for i in range(10):
                print('nop')
        if platform.system() == 'Linux':
            infos = info()
            assert '__linux=' in infos
            assert '__linux=0=0' not in infos
        else:
            infos = info(env={**os.environ, 'CONDA_SUBDIR': 'linux-64'})
            assert '__linux=0=0' in infos