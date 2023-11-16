import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest
from pytest_pyodide import run_in_pyodide, spawn_web_server
cpver = f'cp{sys.version_info.major}{sys.version_info.minor}'
WHEEL_BASE = None

@pytest.fixture
def wheel_base(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    with TemporaryDirectory() as tmpdirname:
        global WHEEL_BASE
        WHEEL_BASE = Path(tmpdirname).absolute()
        import site
        monkeypatch.setattr(site, 'getsitepackages', lambda : [WHEEL_BASE], raising=False)
        try:
            yield
        finally:
            WHEEL_BASE = None

@pytest.fixture
def selenium_standalone_micropip(selenium_standalone):
    if False:
        print('Hello World!')
    "Import micropip before entering test so that global initialization of\n    micropip doesn't count towards hiwire refcount.\n    "
    selenium_standalone.run_js('\n        await pyodide.loadPackage("micropip");\n        pyodide.runPython("import micropip");\n        ')
    yield selenium_standalone
SNOWBALL_WHEEL = 'snowballstemmer-2.0.0-py2.py3-none-any.whl'

def test_install_simple(selenium_standalone_micropip):
    if False:
        while True:
            i = 10
    selenium = selenium_standalone_micropip
    assert selenium.run_js("\n            return await pyodide.runPythonAsync(`\n                import os\n                import micropip\n                from pyodide.ffi import to_js\n                # Package 'pyodide-micropip-test' has dependency on 'snowballstemmer'\n                # It is used to test markers support\n                await micropip.install('pyodide-micropip-test')\n                import snowballstemmer\n                stemmer = snowballstemmer.stemmer('english')\n                to_js(stemmer.stemWords('go going goes gone'.split()))\n            `);\n            ") == ['go', 'go', 'goe', 'gone']

@pytest.mark.parametrize('base_url', ["'{base_url}'", "'.'"])
def test_install_custom_url(selenium_standalone_micropip, base_url):
    if False:
        for i in range(10):
            print('nop')
    selenium = selenium_standalone_micropip
    with spawn_web_server(Path(__file__).parent / 'test') as server:
        (server_hostname, server_port, _) = server
        base_url = f'http://{server_hostname}:{server_port}/'
        url = base_url + SNOWBALL_WHEEL
        selenium.run_js(f"\n            await pyodide.runPythonAsync(`\n                import micropip\n                await micropip.install('{url}')\n                import snowballstemmer\n            `);\n            ")

@pytest.mark.xfail_browsers(chrome='node only', firefox='node only')
def test_install_file_protocol_node(selenium_standalone_micropip):
    if False:
        i = 10
        return i + 15
    selenium = selenium_standalone_micropip
    from conftest import DIST_PATH
    pyparsing_wheel_name = list(DIST_PATH.glob('pyparsing*.whl'))[0].name
    selenium.run_js(f"\n        await pyodide.runPythonAsync(`\n            import micropip\n            await micropip.install('file:{pyparsing_wheel_name}')\n            import pyparsing\n        `);\n        ")

def test_install_different_version(selenium_standalone_micropip):
    if False:
        return 10
    selenium = selenium_standalone_micropip
    selenium.run_js('\n        await pyodide.runPythonAsync(`\n            import micropip\n            await micropip.install(\n                "https://files.pythonhosted.org/packages/89/06/2c2d3034b4d6bf22f2a4ae546d16925898658a33b4400cfb7e2c1e2871a3/pytz-2020.5-py2.py3-none-any.whl"\n            );\n        `);\n        ')
    selenium.run_js('\n        await pyodide.runPythonAsync(`\n            import pytz\n            assert pytz.__version__ == "2020.5"\n        `);\n        ')

def test_install_different_version2(selenium_standalone_micropip):
    if False:
        for i in range(10):
            print('nop')
    selenium = selenium_standalone_micropip
    selenium.run_js('\n        await pyodide.runPythonAsync(`\n            import micropip\n            await micropip.install(\n                "pytz == 2020.5"\n            );\n        `);\n        ')
    selenium.run_js('\n        await pyodide.runPythonAsync(`\n            import pytz\n            assert pytz.__version__ == "2020.5"\n        `);\n        ')

@pytest.mark.parametrize('jinja2', ['jinja2', 'Jinja2'])
def test_install_mixed_case2(selenium_standalone_micropip, jinja2):
    if False:
        return 10
    selenium = selenium_standalone_micropip
    selenium.run_js(f'\n        await pyodide.loadPackage("micropip");\n        await pyodide.runPythonAsync(`\n            import micropip\n            await micropip.install("{jinja2}")\n            import jinja2\n        `);\n        ')

def test_list_load_package_from_url(selenium_standalone_micropip):
    if False:
        return 10
    with spawn_web_server(Path(__file__).parent / 'test') as server:
        (server_hostname, server_port, _) = server
        base_url = f'http://{server_hostname}:{server_port}/'
        url = base_url + SNOWBALL_WHEEL
        selenium = selenium_standalone_micropip
        selenium.run_js(f'\n            await pyodide.loadPackage({url!r});\n            await pyodide.runPythonAsync(`\n                import micropip\n                assert "snowballstemmer" in micropip.list()\n            `);\n            ')

def test_list_pyodide_package(selenium_standalone_micropip):
    if False:
        for i in range(10):
            print('nop')
    selenium = selenium_standalone_micropip
    selenium.run_js('\n        await pyodide.runPythonAsync(`\n            import micropip\n            await micropip.install(\n                "regex"\n            );\n        `);\n        ')
    selenium.run_js('\n        await pyodide.runPythonAsync(`\n            import micropip\n            pkgs = micropip.list()\n            assert "regex" in pkgs\n            assert pkgs["regex"].source.lower() == "pyodide"\n        `);\n        ')

def test_list_loaded_from_js(selenium_standalone_micropip):
    if False:
        i = 10
        return i + 15
    selenium = selenium_standalone_micropip
    selenium.run_js('\n        await pyodide.loadPackage("regex");\n        await pyodide.runPythonAsync(`\n            import micropip\n            pkgs = micropip.list()\n            assert "regex" in pkgs\n            assert pkgs["regex"].source.lower() == "pyodide"\n        `);\n        ')

def test_emfs(selenium_standalone_micropip):
    if False:
        i = 10
        return i + 15
    with spawn_web_server(Path(__file__).parent / 'test') as server:
        (server_hostname, server_port, _) = server
        url = f'http://{server_hostname}:{server_port}/'

        @run_in_pyodide(packages=['micropip'])
        async def run_test(selenium, url, wheel_name):
            import micropip
            from pyodide.http import pyfetch
            resp = await pyfetch(url + wheel_name)
            await resp._into_file(open(wheel_name, 'wb'))
            await micropip.install('emfs:' + wheel_name)
            import snowballstemmer
            stemmer = snowballstemmer.stemmer('english')
            assert stemmer.stemWords('go going goes gone'.split()) == ['go', 'go', 'goe', 'gone']
        run_test(selenium_standalone_micropip, url, SNOWBALL_WHEEL)