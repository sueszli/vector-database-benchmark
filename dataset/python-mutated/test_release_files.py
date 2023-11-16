import pytest
import hypothesistooling as tools
from hypothesistooling import releasemanagement as rm
from hypothesistooling.projects import hypothesispython as hp

@pytest.mark.parametrize('project', tools.all_projects())
def test_release_file_exists_and_is_valid(project):
    if False:
        for i in range(10):
            print('nop')
    if project.has_source_changes():
        assert project.has_release(), 'There are source changes but no RELEASE.rst. Please create one to describe your changes.'
        rm.parse_release_file(project.RELEASE_FILE)

@pytest.mark.skipif(not hp.has_release(), reason='Checking that release')
def test_release_file_has_no_merge_conflicts():
    if False:
        while True:
            i = 10
    (_, message) = rm.parse_release_file(hp.RELEASE_FILE)
    assert '<<<' not in message, 'Merge conflict in RELEASE.rst'
    if message in {hp.get_autoupdate_message(x).strip() for x in (True, False)}:
        return
    (_, *recent_changes, _) = hp.CHANGELOG_ANCHOR.split(hp.changelog(), maxsplit=12)
    for entry in recent_changes:
        (_, version, old_msg) = (x.strip() for x in hp.CHANGELOG_BORDER.split(entry))
        assert message not in old_msg, f'Release notes already published for {version}'
        assert old_msg not in message, f'Copied {version} release notes - merge error?'