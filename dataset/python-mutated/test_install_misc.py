import pytest
from .conftest import DEFAULT_PRIVATE_PYPI_SERVER

@pytest.mark.urls
@pytest.mark.extras
@pytest.mark.install
def test_install_uri_with_extras(pipenv_instance_pypi):
    if False:
        return 10
    server = DEFAULT_PRIVATE_PYPI_SERVER.replace('/simple', '')
    file_uri = f'{server}/packages/plette/plette-0.2.2-py2.py3-none-any.whl'
    with pipenv_instance_pypi() as p:
        with open(p.pipfile_path, 'w') as f:
            contents = f'\n[[source]]\nurl = "{p.index_url}"\nverify_ssl = false\nname = "testindex"\n\n[packages]\nplette = {{file = "{file_uri}", extras = ["validation"]}}\n'
            f.write(contents)
        c = p.pipenv('install')
        assert c.returncode == 0
        assert 'plette' in p.lockfile['default']
        assert 'cerberus' in p.lockfile['default']