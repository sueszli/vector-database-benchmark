from pathlib import Path
from IPython.utils.tempdir import NamedFileInTemporaryDirectory
from IPython.utils.tempdir import TemporaryWorkingDirectory

def test_named_file_in_temporary_directory():
    if False:
        i = 10
        return i + 15
    with NamedFileInTemporaryDirectory('filename') as file:
        name = file.name
        assert not file.closed
        assert Path(name).exists()
        file.write(b'test')
    assert file.closed
    assert not Path(name).exists()

def test_temporary_working_directory():
    if False:
        return 10
    with TemporaryWorkingDirectory() as directory:
        directory_path = Path(directory).resolve()
        assert directory_path.exists()
        assert Path.cwd().resolve() == directory_path
    assert not directory_path.exists()
    assert Path.cwd().resolve() != directory_path