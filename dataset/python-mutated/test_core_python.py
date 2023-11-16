from functools import cache
from pathlib import Path
from textwrap import dedent
from typing import Any
import pytest
import yaml
from yaml import CLoader as Loader

def filter_info(info: dict[str, Any], browser: str) -> dict[str, Any]:
    if False:
        print('Hello World!')
    suffix = '-' + browser
    result = {key.removesuffix(suffix): value for (key, value) in info.items()}
    if 'skip' in info and f'skip{suffix}' in info:
        result['skip'] = info['skip'] + info[f'skip{suffix}']
    return result

def possibly_skip_test(request: pytest.FixtureRequest, info: dict[str, Any]) -> dict[str, Any]:
    if False:
        print('Hello World!')
    if 'segfault' in info:
        pytest.skip(f"known segfault: {info['segfault']}")
    if 'xfail' in info:
        reason = info['xfail']
        if request.config.option.run_xfail:
            request.applymarker(pytest.mark.xfail(run=False, reason=f'known failure: {reason}'))
        else:
            pytest.xfail(f'known failure: {reason}')
    return info

def test_cpython_core(main_test, selenium, request):
    if False:
        i = 10
        return i + 15
    [name, info] = main_test
    info = filter_info(info, selenium.browser)
    possibly_skip_test(request, info)
    ignore_tests = info.get('skip', [])
    if not isinstance(ignore_tests, list):
        raise Exception("Invalid python_tests.yaml entry: 'skip' should be a list")
    selenium.load_package(['distutils', 'test'])
    try:
        selenium.run(dedent(f'''\n            import platform\n            from test import libregrtest\n\n            platform.platform(aliased=True)\n            import _testcapi\n            if hasattr(_testcapi, "raise_SIGINT_then_send_None"):\n                # This uses raise() which doesn't work.\n                del _testcapi.raise_SIGINT_then_send_None\n\n            try:\n                libregrtest.main(["{name}"], ignore_tests={ignore_tests}, verbose=True, verbose3=True)\n            except SystemExit as e:\n                if e.code != 0:\n                    raise RuntimeError(f"Failed with code: {{e.code}}")\n            '''))
    except selenium.JavascriptException:
        print(selenium.logs)
        raise

def get_test_info(test: dict[str, Any] | str) -> tuple[str, dict[str, Any]]:
    if False:
        return 10
    if isinstance(test, dict):
        (name, info) = next(iter(test.items()))
    else:
        name = test
        info = {}
    return (name, info)

@cache
def get_tests() -> list[tuple[str, dict[str, Any]]]:
    if False:
        return 10
    with open(Path(__file__).parent / 'python_tests.yaml') as file:
        data = yaml.load(file, Loader)
    return [get_test_info(test) for test in data]

def pytest_generate_tests(metafunc):
    if False:
        i = 10
        return i + 15
    if 'main_test' in metafunc.fixturenames:
        tests = get_tests()
        metafunc.parametrize('main_test', [pytest.param(t, marks=pytest.mark.requires_dynamic_linking) for t in tests], ids=[t[0] for t in tests])