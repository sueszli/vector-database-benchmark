import os
import logging
from edk2toolext.environment.plugintypes.uefi_build_plugin import IUefiBuildPlugin
from edk2toolext.environment import shell_environment

class LinuxGcc5ToolChain(IUefiBuildPlugin):

    def do_post_build(self, thebuilder):
        if False:
            while True:
                i = 10
        return 0

    def do_pre_build(self, thebuilder):
        if False:
            return 10
        self.Logger = logging.getLogger('LinuxGcc5ToolChain')
        if thebuilder.env.GetValue('TOOL_CHAIN_TAG') == 'GCC5':
            ret = self._check_aarch64()
            if ret != 0:
                self.Logger.critical('Failed in check aarch64')
                return ret
            ret = self._check_arm()
            if ret != 0:
                self.Logger.critical('Failed in check arm')
                return ret
            ret = self._check_riscv64()
            if ret != 0:
                self.Logger.critical('Failed in check riscv64')
                return ret
            ret = self._check_loongarch64()
            if ret != 0:
                self.Logger.critical('Failed in check loongarch64')
                return ret
        return 0

    def _check_arm(self):
        if False:
            print('Hello World!')
        if shell_environment.GetEnvironment().get_shell_var('GCC5_ARM_PREFIX') is not None:
            self.Logger.info('GCC5_ARM_PREFIX is already set.')
        else:
            install_path = shell_environment.GetEnvironment().get_shell_var('GCC5_ARM_INSTALL')
            if install_path is None:
                return 0
            prefix = os.path.join(install_path, 'bin', 'arm-none-linux-gnueabihf-')
            shell_environment.GetEnvironment().set_shell_var('GCC5_ARM_PREFIX', prefix)
        if not os.path.exists(shell_environment.GetEnvironment().get_shell_var('GCC5_ARM_PREFIX') + 'gcc'):
            self.Logger.error('Path for GCC5_ARM_PREFIX toolchain is invalid')
            return -2
        return 0

    def _check_aarch64(self):
        if False:
            return 10
        if shell_environment.GetEnvironment().get_shell_var('GCC5_AARCH64_PREFIX') is not None:
            self.Logger.info('GCC5_AARCH64_PREFIX is already set.')
        else:
            install_path = shell_environment.GetEnvironment().get_shell_var('GCC5_AARCH64_INSTALL')
            if install_path is None:
                return 0
            prefix = os.path.join(install_path, 'bin', 'aarch64-none-linux-gnu-')
            shell_environment.GetEnvironment().set_shell_var('GCC5_AARCH64_PREFIX', prefix)
        if not os.path.exists(shell_environment.GetEnvironment().get_shell_var('GCC5_AARCH64_PREFIX') + 'gcc'):
            self.Logger.error('Path for GCC5_AARCH64_PREFIX toolchain is invalid')
            return -2
        return 0

    def _check_riscv64(self):
        if False:
            for i in range(10):
                print('nop')
        install_path = shell_environment.GetEnvironment().get_shell_var('GCC5_RISCV64_INSTALL')
        if install_path is None:
            return 0
        if shell_environment.GetEnvironment().get_shell_var('GCC5_RISCV64_PREFIX') is not None:
            self.Logger.info('GCC5_RISCV64_PREFIX is already set.')
        else:
            prefix = os.path.join(install_path, 'bin', 'riscv64-unknown-elf-')
            shell_environment.GetEnvironment().set_shell_var('GCC5_RISCV64_PREFIX', prefix)
        if not os.path.exists(shell_environment.GetEnvironment().get_shell_var('GCC5_RISCV64_PREFIX') + 'gcc'):
            self.Logger.error('Path for GCC5_RISCV64_PREFIX toolchain is invalid')
            return -2
        if shell_environment.GetEnvironment().get_shell_var('LD_LIBRARY_PATH') is not None:
            self.Logger.info('LD_LIBRARY_PATH is already set.')
        prefix = os.path.join(install_path, 'lib')
        shell_environment.GetEnvironment().set_shell_var('LD_LIBRARY_PATH', prefix)
        return 0

    def _check_loongarch64(self):
        if False:
            while True:
                i = 10
        if shell_environment.GetEnvironment().get_shell_var('GCC5_LOONGARCH64_PREFIX') is not None:
            self.Logger.info('GCC5_LOONGARCH64_PREFIX is already set.')
        else:
            install_path = shell_environment.GetEnvironment().get_shell_var('GCC5_LOONGARCH64_INSTALL')
            if install_path is None:
                return 0
            prefix = os.path.join(install_path, 'bin', 'loongarch64-unknown-linux-gnu-')
            shell_environment.GetEnvironment().set_shell_var('GCC5_LOONGARCH64_PREFIX', prefix)
        if not os.path.exists(shell_environment.GetEnvironment().get_shell_var('GCC5_LOONGARCH64_PREFIX') + 'gcc'):
            self.Logger.error('Path for GCC5_LOONGARCH64_PREFIX toolchain is invalid')
            return -2
        return 0