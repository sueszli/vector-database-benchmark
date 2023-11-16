import pytest

@pytest.mark.parametrize('a', ['qwe/\\abc'])
def test_fixture(tmp_path, a):
    if False:
        while True:
            i = 10
    assert tmp_path.is_dir()
    assert list(tmp_path.iterdir()) == []