from __future__ import annotations
from ansible.cli.galaxy import _display_role

def test_display_role(mocker, capsys):
    if False:
        i = 10
        return i + 15
    mocked_galaxy_role = mocker.Mock(install_info=None)
    mocked_galaxy_role.name = 'testrole'
    _display_role(mocked_galaxy_role)
    (out, err) = capsys.readouterr()
    out_lines = out.splitlines()
    assert out_lines[0] == '- testrole, (unknown version)'

def test_display_role_known_version(mocker, capsys):
    if False:
        print('Hello World!')
    mocked_galaxy_role = mocker.Mock(install_info={'version': '1.0.0'})
    mocked_galaxy_role.name = 'testrole'
    _display_role(mocked_galaxy_role)
    (out, err) = capsys.readouterr()
    out_lines = out.splitlines()
    assert out_lines[0] == '- testrole, 1.0.0'