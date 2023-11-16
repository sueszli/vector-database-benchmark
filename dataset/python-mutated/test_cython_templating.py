import pathlib
import pytest
import sklearn

def test_files_generated_by_templates_are_git_ignored():
    if False:
        i = 10
        return i + 15
    'Check the consistence of the files generated from template files.'
    gitignore_file = pathlib.Path(sklearn.__file__).parent.parent / '.gitignore'
    if not gitignore_file.exists():
        pytest.skip('Tests are not run from the source folder')
    base_dir = pathlib.Path(sklearn.__file__).parent
    ignored_files = gitignore_file.read_text().split('\n')
    ignored_files = [pathlib.Path(line) for line in ignored_files]
    for filename in base_dir.glob('**/*.tp'):
        filename = filename.relative_to(base_dir.parent)
        filename_wo_tempita_suffix = filename.with_suffix('')
        assert filename_wo_tempita_suffix in ignored_files