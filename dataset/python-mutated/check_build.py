import os
import stat
import subprocess
import sys
import unittest
from functools import partial
from . import BaseTest

class TestBuild(BaseTest):

    def test_exe(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        from kitty.constants import kitten_exe, kitty_exe, str_version
        exe = kitty_exe()
        self.assertTrue(os.access(exe, os.X_OK))
        self.assertTrue(os.path.isfile(exe))
        self.assertIn('kitty', os.path.basename(exe))
        exe = kitten_exe()
        self.assertTrue(os.access(exe, os.X_OK))
        self.assertTrue(os.path.isfile(exe))
        self.assertIn(str_version, subprocess.check_output([exe, '--version']).decode())

    def test_loading_extensions(self) -> None:
        if False:
            while True:
                i = 10
        import kitty.fast_data_types as fdt
        from kittens.transfer import rsync
        del fdt, rsync

    def test_loading_shaders(self) -> None:
        if False:
            return 10
        from kitty.shaders import Program
        for name in 'cell border bgimage tint graphics'.split():
            Program(name)

    def test_glfw_modules(self) -> None:
        if False:
            print('Hello World!')
        from kitty.constants import glfw_path, is_macos
        linux_backends = ['x11']
        if not self.is_ci:
            linux_backends.append('wayland')
        modules = ['cocoa'] if is_macos else linux_backends
        for name in modules:
            path = glfw_path(name)
            self.assertTrue(os.path.isfile(path), f'{path} is not a file')
            self.assertTrue(os.access(path, os.X_OK), f'{path} is not executable')

    def test_all_kitten_names(self) -> None:
        if False:
            i = 10
            return i + 15
        from kittens.runner import all_kitten_names
        names = all_kitten_names()
        self.assertIn('diff', names)
        self.assertIn('hints', names)
        self.assertGreater(len(names), 8)

    def test_filesystem_locations(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        from kitty.constants import local_docs, logo_png_file, shell_integration_dir, terminfo_dir
        zsh = os.path.join(shell_integration_dir, 'zsh')
        self.assertTrue(os.path.isdir(terminfo_dir), f'Terminfo dir: {terminfo_dir}')
        self.assertTrue(os.path.exists(logo_png_file), f'Logo file: {logo_png_file}')
        self.assertTrue(os.path.exists(zsh), f'Shell integration: {zsh}')

        def is_executable(x):
            if False:
                i = 10
                return i + 15
            mode = os.stat(x).st_mode
            q = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            return mode & q == q
        for x in ('kitty', 'kitten'):
            x = os.path.join(shell_integration_dir, 'ssh', x)
            self.assertTrue(is_executable(x), f'{x} is not executable')
        if getattr(sys, 'frozen', False):
            self.assertTrue(os.path.isdir(local_docs()), f'Local docs: {local_docs()}')

    def test_ca_certificates(self):
        if False:
            return 10
        import ssl
        if not getattr(sys, 'frozen', False):
            self.skipTest('CA certificates are only tested on frozen builds')
        c = ssl.create_default_context()
        self.assertGreater(c.cert_store_stats()['x509_ca'], 2)

    def test_docs_url(self):
        if False:
            return 10
        from kitty.constants import website_url
        from kitty.utils import docs_url

        def run_tests(p, base, suffix='.html'):
            if False:
                while True:
                    i = 10

            def t(x, e):
                if False:
                    for i in range(10):
                        print('nop')
                self.ae(p(x), base + e)
            t('', 'index.html' if suffix == '.html' else '')
            t('conf', f'conf{suffix}')
            t('kittens/ssh#frag', f'kittens/ssh{suffix}#frag')
            t('#ref=confloc', f'conf{suffix}#confloc')
            t('#ref=conf-kitty-fonts', f'conf{suffix}#conf-kitty-fonts')
            t('#ref=conf-kitten-ssh-xxx', f'kittens/ssh{suffix}#conf-kitten-ssh-xxx')
            t('#ref=at_close_tab', f'remote-control{suffix}#at-close-tab')
            t('#ref=at-close-tab', f'remote-control{suffix}#at-close-tab')
            t('#ref=action-copy', f'actions{suffix}#copy')
            t('#ref=doc-/marks', f'marks{suffix}')
        run_tests(partial(docs_url, local_docs_root='/docs'), 'file:///docs/')
        w = website_url()
        run_tests(partial(docs_url, local_docs_root=None), w, '/')
        self.ae(docs_url('#ref=issues-123'), 'https://github.com/kovidgoyal/kitty/issues/123')

    def test_launcher_ensures_stdio(self):
        if False:
            print('Hello World!')
        import subprocess
        from kitty.constants import kitty_exe
        exe = kitty_exe()
        cp = subprocess.run([exe, '+runpy', f"import os, sys\nif sys.stdin:\n    os.close(sys.stdin.fileno())\nif sys.stdout:\n    os.close(sys.stdout.fileno())\nif sys.stderr:\n    os.close(sys.stderr.fileno())\nos.execlp({exe!r}, 'kitty', '+runpy', 'import sys; raise SystemExit(1 if sys.stdout is None or sys.stdin is None or sys.stderr is None else 0)')\n"])
        self.assertEqual(cp.returncode, 0)

def main() -> None:
    if False:
        while True:
            i = 10
    tests = unittest.defaultTestLoader.loadTestsFromTestCase(TestBuild)
    r = unittest.TextTestRunner(verbosity=4)
    result = r.run(tests)
    if result.errors or result.failures:
        raise SystemExit(1)