import os
import pytest
from transformers.dynamic_module_utils import get_imports
TOP_LEVEL_IMPORT = '\nimport os\n'
IMPORT_IN_FUNCTION = '\ndef foo():\n    import os\n    return False\n'
DEEPLY_NESTED_IMPORT = '\ndef foo():\n    def bar():\n        if True:\n            import os\n        return False\n    return bar()\n'
TOP_LEVEL_TRY_IMPORT = '\nimport os\n\ntry:\n    import bar\nexcept ImportError:\n    raise ValueError()\n'
TRY_IMPORT_IN_FUNCTION = '\nimport os\n\ndef foo():\n    try:\n        import bar\n    except ImportError:\n        raise ValueError()\n'
MULTIPLE_EXCEPTS_IMPORT = '\nimport os\n\ntry:\n    import bar\nexcept (ImportError, AttributeError):\n    raise ValueError()\n'
EXCEPT_AS_IMPORT = '\nimport os\n\ntry:\n    import bar\nexcept ImportError as e:\n    raise ValueError()\n'
GENERIC_EXCEPT_IMPORT = '\nimport os\n\ntry:\n    import bar\nexcept:\n    raise ValueError()\n'
MULTILINE_TRY_IMPORT = '\nimport os\n\ntry:\n    import bar\n    import baz\nexcept ImportError:\n    raise ValueError()\n'
MULTILINE_BOTH_IMPORT = '\nimport os\n\ntry:\n    import bar\n    import baz\nexcept ImportError:\n    x = 1\n    raise ValueError()\n'
CASES = [TOP_LEVEL_IMPORT, IMPORT_IN_FUNCTION, DEEPLY_NESTED_IMPORT, TOP_LEVEL_TRY_IMPORT, GENERIC_EXCEPT_IMPORT, MULTILINE_TRY_IMPORT, MULTILINE_BOTH_IMPORT, MULTIPLE_EXCEPTS_IMPORT, EXCEPT_AS_IMPORT, TRY_IMPORT_IN_FUNCTION]

@pytest.mark.parametrize('case', CASES)
def test_import_parsing(tmp_path, case):
    if False:
        while True:
            i = 10
    tmp_file_path = os.path.join(tmp_path, 'test_file.py')
    with open(tmp_file_path, 'w') as _tmp_file:
        _tmp_file.write(case)
    parsed_imports = get_imports(tmp_file_path)
    assert parsed_imports == ['os']