from PyInstaller.utils.tests import requires

@requires('six >= 1.0')
def test_six_moves(pyi_builder):
    if False:
        i = 10
        return i + 15
    pyi_builder.test_source('\n        from six.moves import UserList\n        UserList\n        ')

@requires('six >= 1.0')
def test_six_moves_2nd_run(pyi_builder):
    if False:
        i = 10
        return i + 15
    return test_six_moves(pyi_builder)