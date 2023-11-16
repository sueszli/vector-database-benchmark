"""A set of test cases for the wemake isort profile.

Snippets are taken directly from the wemake-python-styleguide project here:
https://github.com/wemake-services/wemake-python-styleguide
"""
from functools import partial
from ..utils import isort_test
wemake_isort_test = partial(isort_test, profile='wemake', known_first_party=['wemake_python_styleguide'])

def test_wemake_snippet_one():
    if False:
        return 10
    wemake_isort_test('\nimport ast\nimport tokenize\nimport traceback\nfrom typing import ClassVar, Iterator, Sequence, Type\n\nfrom flake8.options.manager import OptionManager\nfrom typing_extensions import final\n\nfrom wemake_python_styleguide import constants, types\nfrom wemake_python_styleguide import version as pkg_version\nfrom wemake_python_styleguide.options.config import Configuration\nfrom wemake_python_styleguide.options.validation import validate_options\nfrom wemake_python_styleguide.presets.types import file_tokens as tokens_preset\nfrom wemake_python_styleguide.presets.types import filename as filename_preset\nfrom wemake_python_styleguide.presets.types import tree as tree_preset\nfrom wemake_python_styleguide.transformations.ast_tree import transform\nfrom wemake_python_styleguide.violations import system\nfrom wemake_python_styleguide.visitors import base\n\nVisitorClass = Type[base.BaseVisitor]\n')

def test_wemake_snippet_two():
    if False:
        for i in range(10):
            print('nop')
    wemake_isort_test("\nfrom collections import defaultdict\nfrom typing import ClassVar, DefaultDict, List\n\nfrom flake8.formatting.base import BaseFormatter\nfrom flake8.statistics import Statistics\nfrom flake8.style_guide import Violation\nfrom pygments import highlight\nfrom pygments.formatters import TerminalFormatter\nfrom pygments.lexers import PythonLexer\nfrom typing_extensions import Final\n\nfrom wemake_python_styleguide.version import pkg_version\n\n#: That url is generated and hosted by Sphinx.\nDOCS_URL_TEMPLATE: Final = (\n    'https://wemake-python-stylegui.de/en/{0}/pages/usage/violations/'\n)\n")

def test_wemake_snippet_three():
    if False:
        for i in range(10):
            print('nop')
    wemake_isort_test('\nimport ast\n\nfrom pep8ext_naming import NamingChecker\nfrom typing_extensions import final\n\nfrom wemake_python_styleguide.transformations.ast.bugfixes import (\n    fix_async_offset,\n    fix_line_number,\n)\nfrom wemake_python_styleguide.transformations.ast.enhancements import (\n    set_if_chain,\n    set_node_context,\n)\n\n\n@final\nclass _ClassVisitor(ast.NodeVisitor): ...\n')