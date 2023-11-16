import pytest
from pdm.models.setup import Setup

@pytest.mark.parametrize('content, result', [('[metadata]\nname = foo\nversion = 0.1.0\n', Setup('foo', '0.1.0')), ('[metadata]\nname = foo\nversion = attr:foo.__version__\n', Setup('foo', '0.0.0')), ('[metadata]\nname = foo\nversion = 0.1.0\n\n[options]\npython_requires = >=3.6\ninstall_requires =\n    click\n    requests\n[options.extras_require]\ntui =\n    rich\n', Setup('foo', '0.1.0', ['click', 'requests'], {'tui': ['rich']}, '>=3.6'))])
def test_parse_setup_cfg(content, result, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    tmp_path.joinpath('setup.cfg').write_text(content)
    assert Setup.from_directory(tmp_path) == result

@pytest.mark.parametrize('content,result', [('from setuptools import setup\n\nsetup(name="foo", version="0.1.0")\n', Setup('foo', '0.1.0')), ('import setuptools\n\nsetuptools.setup(name="foo", version="0.1.0")\n', Setup('foo', '0.1.0')), ('from setuptools import setup\n\nkwargs = {"name": "foo", "version": "0.1.0"}\nsetup(**kwargs)\n', Setup('foo', '0.1.0')), ('from setuptools import setup\nname = \'foo\'\nsetup(name=name, version="0.1.0")\n', Setup('foo', '0.1.0')), ('from setuptools import setup\n\nsetup(name="foo", version="0.1.0", install_requires=[\'click\', \'requests\'],\n      python_requires=\'>=3.6\', extras_require={\'tui\': [\'rich\']})\n', Setup('foo', '0.1.0', ['click', 'requests'], {'tui': ['rich']}, '>=3.6')), ('from pathlib import Path\nfrom setuptools import setup\n\nversion = Path(\'__version__.py\').read_text().strip()\n\nsetup(name="foo", version=version)\n', Setup('foo', '0.0.0'))])
def test_parse_setup_py(content, result, tmp_path):
    if False:
        while True:
            i = 10
    tmp_path.joinpath('setup.py').write_text(content)
    assert Setup.from_directory(tmp_path) == result

def test_parse_pyproject_toml(tmp_path):
    if False:
        while True:
            i = 10
    content = '[project]\nname = "foo"\nversion = "0.1.0"\nrequires-python = ">=3.6"\ndependencies = ["click", "requests"]\n\n[project.optional-dependencies]\ntui = ["rich"]\n'
    tmp_path.joinpath('pyproject.toml').write_text(content)
    result = Setup('foo', '0.1.0', ['click', 'requests'], {'tui': ['rich']}, '>=3.6')
    assert Setup.from_directory(tmp_path) == result