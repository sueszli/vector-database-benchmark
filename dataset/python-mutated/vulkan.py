import os
from glob import glob
import shutil
import subprocess
import filecmp

class VulkanUtils:
    __vk_icd_dirs = ['/usr/share/vulkan', '/etc/vulkan', '/usr/local/share/vulkan', '/usr/local/etc/vulkan']
    if 'FLATPAK_ID' in os.environ:
        __vk_icd_dirs += ['/usr/lib/x86_64-linux-gnu/GL/vulkan', '/usr/lib/i386-linux-gnu/GL/vulkan']

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.loaders = self.__get_vk_icd_loaders()

    def __get_vk_icd_loaders(self):
        if False:
            i = 10
            return i + 15
        loaders = {'nvidia': [], 'amd': [], 'intel': []}
        for _dir in self.__vk_icd_dirs:
            _files = glob(f'{_dir}/icd.d/*.json', recursive=True)
            for file in _files:
                if 'nvidia' in file.lower():
                    should_skip = False
                    for nvidia_loader in loaders['nvidia']:
                        if filecmp.cmp(nvidia_loader, file):
                            should_skip = True
                            continue
                    if not should_skip:
                        loaders['nvidia'] += [file]
                elif 'amd' in file.lower() or 'radeon' in file.lower():
                    loaders['amd'] += [file]
                elif 'intel' in file.lower():
                    loaders['intel'] += [file]
        return loaders

    def get_vk_icd(self, vendor: str, as_string=False):
        if False:
            return 10
        vendors = ['nvidia', 'amd', 'intel']
        icd = []
        if vendor in vendors:
            icd = self.loaders[vendor]
        if as_string:
            icd = ':'.join(icd)
        return icd

    @staticmethod
    def check_support():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def test_vulkan():
        if False:
            while True:
                i = 10
        if shutil.which('vulkaninfo') is None:
            return 'vulkaninfo tool not found'
        res = subprocess.Popen('vulkaninfo', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).communicate()[0].decode('utf-8')
        return res