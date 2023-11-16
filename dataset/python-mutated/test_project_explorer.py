"""
Tests for explorer.py
"""
import os
import os.path as osp
import pytest
from spyder.plugins.projects.widgets.main_widget import ProjectExplorerTest
from spyder.py3compat import to_text_string

@pytest.fixture
def project_explorer(qtbot, request, tmpdir):
    if False:
        while True:
            i = 10
    'Setup Project Explorer widget.'
    directory = request.node.get_closest_marker('change_directory')
    if directory:
        project_dir = to_text_string(tmpdir.mkdir('project'))
    else:
        project_dir = None
    project_explorer = ProjectExplorerTest(directory=project_dir)
    qtbot.addWidget(project_explorer)
    return project_explorer

def test_project_explorer(project_explorer):
    if False:
        i = 10
        return i + 15
    'Run ProjectExplorerTest.'
    project_explorer.resize(640, 480)
    project_explorer.show()
    assert project_explorer

@pytest.mark.change_directory
def test_change_directory_in_project_explorer(project_explorer, qtbot):
    if False:
        while True:
            i = 10
    'Test changing a file from directory in the Project explorer.'
    project = project_explorer
    project_dir = project.directory
    project_dir_tmp = osp.join(project_dir, u'測試')
    project_file = osp.join(project_dir, 'script.py')
    os.mkdir(project_dir_tmp)
    open(project_file, 'w').close()
    project.explorer.treewidget.move(fnames=[osp.join(project_dir, 'script.py')], directory=project_dir_tmp)
    assert osp.isfile(osp.join(project_dir_tmp, 'script.py'))

def test_project_explorer(project_explorer, qtbot):
    if False:
        for i in range(10):
            print('nop')
    'Run project explorer.'
    project = project_explorer
    project.resize(250, 480)
    project.show()
    assert project
if __name__ == '__main__':
    pytest.main()