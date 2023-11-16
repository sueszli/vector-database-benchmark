"""
Tests for projectdialog.py
"""
import os
from unittest.mock import Mock
import pytest
from spyder.plugins.projects.widgets.projectdialog import ProjectDialog
from spyder.plugins.projects.api import EmptyProject

@pytest.fixture
def projects_dialog(qtbot):
    if False:
        return 10
    'Set up ProjectDialog.'
    dlg = ProjectDialog(None, {'Empty project': EmptyProject})
    qtbot.addWidget(dlg)
    dlg.show()
    return dlg

@pytest.mark.skipif(os.name != 'nt', reason='Specific to Windows platform')
def test_projectdialog_location(monkeypatch):
    if False:
        return 10
    'Test that select_location normalizes delimiters and updates the path.'
    dlg = ProjectDialog(None, {'Empty project': EmptyProject})
    mock_getexistingdirectory = Mock()
    monkeypatch.setattr('spyder.plugins.projects.widgets.projectdialog' + '.getexistingdirectory', mock_getexistingdirectory)
    mock_getexistingdirectory.return_value = 'c:\\a/b\\\\c/d'
    dlg.select_location()
    assert dlg.location == 'c:\\a\\b\\c\\d'
    mock_getexistingdirectory.return_value = 'c:\\\\a//b\\\\c//d'
    dlg.select_location()
    assert dlg.location == 'c:\\a\\b\\c\\d'
    mock_getexistingdirectory.return_value = 'c:\\a\\b\\c/d'
    dlg.select_location()
    assert dlg.location == 'c:\\a\\b\\c\\d'
    mock_getexistingdirectory.return_value = 'c:/a/b/c\\d'
    dlg.select_location()
    assert dlg.location == 'c:\\a\\b\\c\\d'
    mock_getexistingdirectory.return_value = 'c:\\\\a\\\\b\\\\c//d'
    dlg.select_location()
    assert dlg.location == 'c:\\a\\b\\c\\d'
    mock_getexistingdirectory.return_value = 'c:\\AaA/bBB1\\\\c-C/d2D'
    dlg.select_location()
    assert dlg.location == 'c:\\AaA\\bBB1\\c-C\\d2D'
    mock_getexistingdirectory.return_value = 'c:\\\\a_a_1//Bbbb\\2345//d-6D'
    dlg.select_location()
    assert dlg.location == 'c:\\a_a_1\\Bbbb\\2345\\d-6D'

def test_directory_validations(projects_dialog, monkeypatch, tmpdir):
    if False:
        while True:
            i = 10
    '\n    Test that we perform appropiate validations before allowing users to\n    create a project in a directory.\n    '
    dlg = projects_dialog
    assert not dlg.button_create.isEnabled()
    assert not dlg.button_create.isDefault()
    dlg.location = str(tmpdir)
    dlg.text_location.setText(str(tmpdir))
    dlg.radio_new_dir.click()
    tmpdir.mkdir('foo')
    dlg.text_project_name.setText('foo')
    assert not dlg.button_create.isEnabled()
    assert not dlg.button_create.isDefault()
    assert dlg.label_information.text() == '\nThis directory already exists!'
    dlg.radio_from_dir.click()
    assert dlg.button_create.isEnabled()
    assert dlg.button_create.isDefault()
    assert dlg.label_information.text() == ''
    folder = tmpdir.mkdir('bar')
    folder.mkdir('.spyproject')
    mock_getexistingdirectory = Mock()
    monkeypatch.setattr('spyder.plugins.projects.widgets.projectdialog' + '.getexistingdirectory', mock_getexistingdirectory)
    mock_getexistingdirectory.return_value = str(folder)
    dlg.select_location()
    assert not dlg.button_create.isEnabled()
    assert not dlg.button_create.isDefault()
    msg = '\nThis directory is already a Spyder project!'
    assert dlg.label_information.text() == msg
if __name__ == '__main__':
    pytest.main()