from os.path import exists
import pytest
from tests.lib import PipTestEnvironment, TestData

@pytest.mark.network
@pytest.mark.xfail(reason='The --build option was removed')
def test_no_clean_option_blocks_cleaning_after_install(script: PipTestEnvironment, data: TestData) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Test --no-clean option blocks cleaning after install\n    '
    build = script.base_path / 'pip-build'
    script.pip('install', '--no-clean', '--no-index', '--build', build, f'--find-links={data.find_links}', 'simple', expect_temp=True, allow_stderr_warning=True)
    assert exists(build)

@pytest.mark.network
def test_pep517_no_legacy_cleanup(script: PipTestEnvironment, data: TestData) -> None:
    if False:
        print('Hello World!')
    'Test a PEP 517 failed build does not attempt a legacy cleanup'
    to_install = data.packages.joinpath('pep517_wrapper_buildsys')
    script.environ['PIP_TEST_FAIL_BUILD_WHEEL'] = '1'
    res = script.pip('install', '-f', data.find_links, to_install, expect_error=True)
    expected = 'Failed building wheel for pep517-wrapper-buildsys'
    assert expected in str(res)
    assert 'setup.py clean' not in str(res)