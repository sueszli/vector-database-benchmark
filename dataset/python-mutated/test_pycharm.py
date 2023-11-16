from functools import partial
from ..utils import isort_test
pycharm_isort_test = partial(isort_test, profile='pycharm')

def test_pycharm_snippet_one():
    if False:
        print('Hello World!')
    pycharm_isort_test('import shutil\nimport sys\nfrom io import StringIO\nfrom pathlib import Path\nfrom typing import (\n    Optional,\n    TextIO,\n    Union,\n    cast\n)\nfrom warnings import warn\n\nfrom isort import core\n\nfrom . import io\nfrom .exceptions import (\n    ExistingSyntaxErrors,\n    FileSkipComment,\n    FileSkipSetting,\n    IntroducedSyntaxErrors\n)\nfrom .format import (\n    ask_whether_to_apply_changes_to_file,\n    create_terminal_printer,\n    show_unified_diff\n)\nfrom .io import Empty\nfrom .place import module as place_module  # noqa: F401\nfrom .place import module_with_reason as place_module_with_reason  # noqa: F401\nfrom .settings import (\n    DEFAULT_CONFIG,\n    Config\n)\n\n\ndef sort_code_string(\n    code: str,\n    extension: Optional[str] = None,\n    config: Config = DEFAULT_CONFIG,\n    file_path: Optional[Path] = None,\n    disregard_skip: bool = False,\n    show_diff: Union[bool, TextIO] = False,\n    **config_kwargs,\n):\n')