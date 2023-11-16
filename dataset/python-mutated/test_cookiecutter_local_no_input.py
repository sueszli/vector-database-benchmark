"""Test cookiecutter for work without any input.

Tests in this file execute `cookiecutter()` with `no_input=True` flag and
verify result with different settings in `cookiecutter.json`.
"""
import os
import textwrap
from pathlib import Path
import pytest
from cookiecutter import main, utils

@pytest.fixture(scope='function')
def remove_additional_dirs(request):
    if False:
        while True:
            i = 10
    'Fixture. Remove special directories which are created during the tests.'

    def fin_remove_additional_dirs():
        if False:
            while True:
                i = 10
        if os.path.isdir('fake-project'):
            utils.rmtree('fake-project')
        if os.path.isdir('fake-project-extra'):
            utils.rmtree('fake-project-extra')
        if os.path.isdir('fake-project-templated'):
            utils.rmtree('fake-project-templated')
        if os.path.isdir('fake-project-dict'):
            utils.rmtree('fake-project-dict')
        if os.path.isdir('fake-tmp'):
            utils.rmtree('fake-tmp')
    request.addfinalizer(fin_remove_additional_dirs)

@pytest.mark.parametrize('path', ['tests/fake-repo-pre/', 'tests/fake-repo-pre'])
@pytest.mark.usefixtures('clean_system', 'remove_additional_dirs')
def test_cookiecutter_no_input_return_project_dir(path):
    if False:
        i = 10
        return i + 15
    'Verify `cookiecutter` create project dir on input with or without slash.'
    project_dir = main.cookiecutter(path, no_input=True)
    assert os.path.isdir('tests/fake-repo-pre/{{cookiecutter.repo_name}}')
    assert not os.path.isdir('tests/fake-repo-pre/fake-project')
    assert os.path.isdir(project_dir)
    assert os.path.isfile('fake-project/README.rst')
    assert not os.path.exists('fake-project/json/')

@pytest.mark.usefixtures('clean_system', 'remove_additional_dirs')
def test_cookiecutter_no_input_extra_context():
    if False:
        while True:
            i = 10
    'Verify `cookiecutter` accept `extra_context` argument.'
    main.cookiecutter('tests/fake-repo-pre', no_input=True, extra_context={'repo_name': 'fake-project-extra'})
    assert os.path.isdir('fake-project-extra')

@pytest.mark.usefixtures('clean_system', 'remove_additional_dirs')
def test_cookiecutter_templated_context():
    if False:
        while True:
            i = 10
    'Verify Jinja2 templating correctly works in `cookiecutter.json` file.'
    main.cookiecutter('tests/fake-repo-tmpl', no_input=True)
    assert os.path.isdir('fake-project-templated')

@pytest.mark.usefixtures('clean_system', 'remove_additional_dirs')
def test_cookiecutter_no_input_return_rendered_file():
    if False:
        return 10
    'Verify Jinja2 templating correctly works in `cookiecutter.json` file.'
    project_dir = main.cookiecutter('tests/fake-repo-pre', no_input=True)
    assert project_dir == os.path.abspath('fake-project')
    content = Path(project_dir, 'README.rst').read_text()
    assert 'Project name: **Fake Project**' in content

@pytest.mark.usefixtures('clean_system', 'remove_additional_dirs')
def test_cookiecutter_dict_values_in_context():
    if False:
        i = 10
        return i + 15
    'Verify configured dictionary from `cookiecutter.json` correctly unpacked.'
    project_dir = main.cookiecutter('tests/fake-repo-dict', no_input=True)
    assert project_dir == os.path.abspath('fake-project-dict')
    content = Path(project_dir, 'README.md').read_text()
    assert content == textwrap.dedent('\n        # README\n\n\n        <dl>\n          <dt>Format name:</dt>\n          <dd>Bitmap</dd>\n\n          <dt>Extension:</dt>\n          <dd>bmp</dd>\n\n          <dt>Applications:</dt>\n          <dd>\n              <ul>\n              <li>Paint</li>\n              <li>GIMP</li>\n              </ul>\n          </dd>\n        </dl>\n\n        <dl>\n          <dt>Format name:</dt>\n          <dd>Portable Network Graphic</dd>\n\n          <dt>Extension:</dt>\n          <dd>png</dd>\n\n          <dt>Applications:</dt>\n          <dd>\n              <ul>\n              <li>GIMP</li>\n              </ul>\n          </dd>\n        </dl>\n\n    ').lstrip()

@pytest.mark.usefixtures('clean_system', 'remove_additional_dirs')
def test_cookiecutter_template_cleanup(mocker):
    if False:
        print('Hello World!')
    'Verify temporary folder for zip unpacking dropped.'
    mocker.patch('tempfile.mkdtemp', return_value='fake-tmp', autospec=True)
    mocker.patch('cookiecutter.utils.prompt_and_delete', return_value=True, autospec=True)
    main.cookiecutter('tests/files/fake-repo-tmpl.zip', no_input=True)
    assert os.path.isdir('fake-project-templated')
    assert os.path.exists('fake-tmp')
    assert not os.path.exists('fake-tmp/fake-repo-tmpl')