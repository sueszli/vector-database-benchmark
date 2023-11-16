from pathlib import Path
from typing import List
from dagster_buildkite.git import ChangedFiles
from dagster_buildkite.package_spec import PackageSpec
from ..python_version import AvailablePythonVersion
from ..step_builder import CommandStepBuilder
from ..utils import CommandStep, is_feature_branch

def skip_if_no_dagster_ui_changes():
    if False:
        i = 10
        return i + 15
    if not is_feature_branch():
        return None
    if any((Path('js_modules') in path.parents for path in ChangedFiles.all)):
        return None
    if not PackageSpec('python_modules/dagster-graphql').skip_reason:
        return None
    return 'No changes that affect the JS webapp'

def build_dagster_ui_steps() -> List[CommandStep]:
    if False:
        while True:
            i = 10
    return [CommandStepBuilder(':typescript: dagster-ui').run('cd js_modules/dagster-ui', 'curl -sL https://deb.nodesource.com/setup_16.x | bash -', 'apt-get -yqq --no-install-recommends install nodejs', 'tox -vv -e py310').on_test_image(AvailablePythonVersion.get_default()).with_skip(skip_if_no_dagster_ui_changes()).build()]