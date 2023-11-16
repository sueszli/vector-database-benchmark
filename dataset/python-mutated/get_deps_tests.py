import contextlib
import io
import os
import textwrap
import unittest
from mkdocs.commands.get_deps import get_deps
from mkdocs.tests.base import tempdir
_projects_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'integration', 'projects.yaml')

class TestGetDeps(unittest.TestCase):

    @contextlib.contextmanager
    def _assert_logs(self, expected):
        if False:
            return 10
        with self.assertLogs('mkdocs.commands.get_deps') as cm:
            yield
        msgs = [f'{r.levelname}:{r.message}' for r in cm.records]
        self.assertEqual('\n'.join(msgs), textwrap.dedent(expected).strip('\n'))

    @tempdir()
    def _test_get_deps(self, tempdir, yml, expected):
        if False:
            print('Hello World!')
        if yml:
            yml = 'site_name: Test\n' + textwrap.dedent(yml)
        projects_path = os.path.join(tempdir, 'projects.yaml')
        with open(projects_path, 'w', encoding='utf-8') as f:
            f.write(yml)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            get_deps(_projects_file_path, projects_path)
        self.assertEqual(buf.getvalue().split(), expected)

    def test_empty_config(self):
        if False:
            i = 10
            return i + 15
        expected_logs = "WARNING:The passed config file doesn't seem to be a mkdocs.yml config file"
        with self._assert_logs(expected_logs):
            self._test_get_deps('', [])

    def test_just_search(self):
        if False:
            while True:
                i = 10
        cfg = '\n            plugins: [search]\n        '
        self._test_get_deps(cfg, ['mkdocs'])

    def test_mkdocs_config(self):
        if False:
            while True:
                i = 10
        cfg = '\n            site_name: MkDocs\n            theme:\n              name: mkdocs\n              locale: en\n            markdown_extensions:\n              - toc:\n                  permalink: \uf0c1\n              - attr_list\n              - def_list\n              - tables\n              - pymdownx.highlight:\n                  use_pygments: false\n              - pymdownx.snippets\n              - pymdownx.superfences\n              - callouts\n              - mdx_gh_links:\n                  user: mkdocs\n                  repo: mkdocs\n              - mkdocs-click\n            plugins:\n              - search\n              - redirects:\n              - autorefs\n              - literate-nav:\n                  nav_file: README.md\n                  implicit_index: true\n              - mkdocstrings:\n                  handlers:\n                      python:\n                          options:\n                              docstring_section_style: list\n        '
        self._test_get_deps(cfg, ['markdown-callouts', 'mdx-gh-links', 'mkdocs', 'mkdocs-autorefs', 'mkdocs-click', 'mkdocs-literate-nav', 'mkdocs-redirects', 'mkdocstrings', 'mkdocstrings-python', 'pymdown-extensions'])

    def test_dict_keys_and_ignores_env(self):
        if False:
            for i in range(10):
                print('nop')
        cfg = '\n            theme:\n              name: material\n            plugins:\n              code-validator:\n                enabled: !ENV [LINT, false]\n            markdown_extensions:\n              pymdownx.emoji:\n                emoji_index: !!python/name:materialx.emoji.twemoji\n                emoji_generator: !!python/name:materialx.emoji.to_svg\n        '
        self._test_get_deps(cfg, ['mkdocs', 'mkdocs-code-validator', 'mkdocs-material', 'pymdown-extensions'])

    def test_theme_precedence(self):
        if False:
            while True:
                i = 10
        cfg = '\n            plugins:\n              - tags\n            theme: material\n        '
        self._test_get_deps(cfg, ['mkdocs', 'mkdocs-material'])
        cfg = '\n            plugins:\n              - material/tags\n        '
        self._test_get_deps(cfg, ['mkdocs', 'mkdocs-material'])
        cfg = '\n            plugins:\n              - tags\n        '
        self._test_get_deps(cfg, ['mkdocs', 'mkdocs-plugin-tags'])

    def test_nonexistent(self):
        if False:
            print('Hello World!')
        cfg = '\n            plugins:\n              - taglttghhmdu\n              - syyisjupkbpo\n              - redirects\n            theme: qndyakplooyh\n            markdown_extensions:\n              - saqdhyndpvpa\n        '
        expected_logs = "\n            WARNING:Theme 'qndyakplooyh' is not provided by any registered project\n            WARNING:Plugin 'syyisjupkbpo' is not provided by any registered project\n            WARNING:Plugin 'taglttghhmdu' is not provided by any registered project\n            WARNING:Extension 'saqdhyndpvpa' is not provided by any registered project\n        "
        with self._assert_logs(expected_logs):
            self._test_get_deps(cfg, ['mkdocs', 'mkdocs-redirects'])

    def test_git_and_shadowed(self):
        if False:
            while True:
                i = 10
        cfg = '\n            theme: bootstrap4\n            plugins: [blog]\n        '
        self._test_get_deps(cfg, ['git+https://github.com/andyoakley/mkdocs-blog', 'mkdocs', 'mkdocs-bootstrap4'])

    def test_multi_theme(self):
        if False:
            for i in range(10):
                print('nop')
        cfg = '\n            theme: minty\n        '
        self._test_get_deps(cfg, ['mkdocs', 'mkdocs-bootswatch'])

    def test_with_locale(self):
        if False:
            print('Hello World!')
        cfg = '\n            theme:\n                name: mkdocs\n                locale: uk\n        '
        self._test_get_deps(cfg, ['mkdocs[i18n]'])