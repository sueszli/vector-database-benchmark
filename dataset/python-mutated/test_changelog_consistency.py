import os
from pathlib import Path
from mongoengine import get_version
from tests import DOCS_DIR

def test_package_version_described_in_changelog():
    if False:
        while True:
            i = 10
    'Ensures that changelog is updated when version is incremented'
    version_str = get_version()
    changelog_content = Path(os.path.join(DOCS_DIR, 'changelog.rst')).read_text()
    assert version_str in changelog_content, 'Version in __init__.py not present in changelog'

def test_package_version_incremented_when_new_version_added_to_changelog():
    if False:
        print('Hello World!')
    'Ensures that changelog is updated when version is incremented'
    version_str = get_version()
    changelog_content = Path(os.path.join(DOCS_DIR, 'changelog.rst')).read_text()

    def find_between(s, start, end):
        if False:
            i = 10
            return i + 15
        return s.split(start)[1].split(end)[0]
    most_recent_version = find_between(changelog_content, start='Changes in ', end='\n')
    assert most_recent_version == version_str