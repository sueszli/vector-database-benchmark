import logging
import sys
import warnings
import pytest
from salt.utils.functools import namespaced_function
from tests.conftest import CODE_DIR
log = logging.getLogger(__name__)

def preserve_context_ids(value):
    if False:
        i = 10
        return i + 15
    return 'preserve_context={}'.format(value)

@pytest.fixture(params=[True, False], ids=preserve_context_ids)
def preserve_context(request):
    if False:
        for i in range(10):
            print('nop')
    return request.param

def test_namespacing(tmp_path, shell):
    if False:
        for i in range(10):
            print('nop')
    pkgpath = tmp_path / 'foopkg'
    mod1_contents = '\n    import json\n    import time\n\n    def func_1():\n        return time.time()\n\n    def main():\n        data = {\n            "func1": func_1(),\n            "module": func_1.__module__,\n            "time_present": "time" in func_1.__globals__\n        }\n        print(json.dumps(data))\n\n    if __name__ == "__main__":\n        main()\n    '
    mod2_contents = '\n    import json\n    from salt.utils.functools import namespaced_function\n    from foopkg.mod1 import func_1\n\n    func_1 = namespaced_function(func_1, globals())\n\n    def main():\n        data = {\n            "func1": func_1(),\n            "module": func_1.__module__,\n            "time_present": "time" in func_1.__globals__\n        }\n        print(json.dumps(data))\n\n    if __name__ == "__main__":\n        main()\n    '
    run1_contents = "\n    import sys\n    sys.path.insert(0, '{}')\n    import foopkg.mod1\n\n    foopkg.mod1.main()\n    ".format(CODE_DIR)
    run2_contents = "\n    import sys\n    sys.path.insert(0, '{}')\n    import foopkg.mod2\n\n    foopkg.mod2.main()\n    ".format(CODE_DIR)
    with pytest.helpers.temp_file('run1.py', contents=run1_contents, directory=tmp_path), pytest.helpers.temp_file('run2.py', contents=run2_contents, directory=tmp_path), pytest.helpers.temp_file('__init__.py', contents='', directory=pkgpath), pytest.helpers.temp_file('mod1.py', mod1_contents, directory=pkgpath), pytest.helpers.temp_file('mod2.py', mod2_contents, directory=pkgpath):
        ret = shell.run(sys.executable, str(tmp_path / 'run1.py'), cwd=str(tmp_path))
        log.warning(ret)
        assert ret.returncode == 0
        assert ret.data['module'] == 'foopkg.mod1'
        assert ret.data['time_present'] is True
        assert isinstance(ret.data['func1'], float)
        ret = shell.run(sys.executable, str(tmp_path / 'run2.py'), cwd=str(tmp_path))
        log.warning(ret)
        assert ret.returncode == 0
        assert ret.data['module'] == 'foopkg.mod2'
        assert isinstance(ret.data['func1'], float)
        assert ret.data['time_present'] is True

def test_deprecated_defaults_kwarg():
    if False:
        for i in range(10):
            print('nop')

    def foo():
        if False:
            return 10
        pass
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        namespaced_function(foo, globals(), defaults={'foo': 1})
    assert str(w[-1].message) == "Passing 'defaults' to 'namespaced_function' is deprecated, slated for removal in 3008.0 (Argon) and no longer does anything for the function being namespaced."

def test_deprecated_preserve_context_kwarg(preserve_context):
    if False:
        print('Hello World!')

    def foo():
        if False:
            i = 10
            return i + 15
        pass
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        namespaced_function(foo, globals(), preserve_context=preserve_context)
    assert str(w[-1].message) == "Passing 'preserve_context' to 'namespaced_function' is deprecated, slated for removal in 3008.0 (Argon) and no longer does anything for the function being namespaced."