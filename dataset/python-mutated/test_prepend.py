import pytest
pytestmark = [pytest.mark.windows_whitelisted]

def test_prepend_issue_27401_makedirs(file, tmp_path):
    if False:
        print('Hello World!')
    "\n    file.prepend but create directories if needed as an option, and create\n    the file if it doesn't exist\n    "
    fname = 'prepend_issue_27401'
    name = tmp_path / fname
    ret = file.prepend(name=str(name), text='cheese', makedirs=True)
    assert ret.result is True
    assert name.is_file()
    assert name.read_text() == 'cheese\n'
    name = tmp_path / 'issue_27401' / fname
    ret = file.prepend(name=str(name), text='cheese', makedirs=True)
    assert ret.result is True
    assert name.is_file()
    assert name.read_text() == 'cheese\n'
    assert name.parent.is_dir()
    name = name.with_name(fname + '2')
    ret = file.prepend(name=str(name), text='cheese', makedirs=False)
    assert ret.result is True
    assert name.is_file()
    assert name.read_text() == 'cheese\n'
    assert name.parent.is_dir()