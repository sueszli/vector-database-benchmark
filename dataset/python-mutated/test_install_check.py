from typing import Iterable
from tests.lib import PipTestEnvironment, create_test_package_with_setup

def assert_contains_expected_lines(string: str, expected_lines: Iterable[str]) -> None:
    if False:
        for i in range(10):
            print('nop')
    for expected_line in expected_lines:
        assert expected_line + '\n' in string

def test_check_install_canonicalization(script: PipTestEnvironment) -> None:
    if False:
        print('Hello World!')
    pkga_path = create_test_package_with_setup(script, name='pkgA', version='1.0', install_requires=['normal-missing', 'SPECIAL.missing'])
    normal_path = create_test_package_with_setup(script, name='normal-missing', version='0.1')
    special_path = create_test_package_with_setup(script, name='SPECIAL.missing', version='0.1')
    result = script.pip('install', '--no-index', pkga_path, '--no-deps')
    assert 'Successfully installed pkgA-1.0' in result.stdout, str(result)
    result = script.pip('install', '--no-index', normal_path, '--quiet', allow_stderr_error=True)
    expected_lines = ['pkga 1.0 requires SPECIAL.missing, which is not installed.']
    assert_contains_expected_lines(result.stderr, expected_lines)
    assert result.returncode == 0
    result = script.pip('install', '--no-index', special_path, '--quiet')
    assert 'requires' not in result.stderr
    assert result.returncode == 0
    result = script.pip('check')
    expected_lines = ['No broken requirements found.']
    assert_contains_expected_lines(result.stdout, expected_lines)
    assert result.returncode == 0

def test_check_install_does_not_warn_for_out_of_graph_issues(script: PipTestEnvironment) -> None:
    if False:
        while True:
            i = 10
    pkg_broken_path = create_test_package_with_setup(script, name='broken', version='1.0', install_requires=['missing', 'conflict < 1.0'])
    pkg_unrelated_path = create_test_package_with_setup(script, name='unrelated', version='1.0')
    pkg_conflict_path = create_test_package_with_setup(script, name='conflict', version='1.0')
    result = script.pip('install', '--no-index', pkg_broken_path, '--no-deps')
    assert 'requires' not in result.stderr
    result = script.pip('install', '--no-index', pkg_conflict_path, allow_stderr_error=True)
    assert_contains_expected_lines(result.stderr, ['broken 1.0 requires missing, which is not installed.', 'broken 1.0 requires conflict<1.0, but you have conflict 1.0 which is incompatible.'])
    result = script.pip('install', '--no-index', pkg_unrelated_path, '--quiet')
    assert 'requires' not in result.stderr
    result = script.pip('check', expect_error=True)
    expected_lines = ['broken 1.0 requires missing, which is not installed.', 'broken 1.0 has requirement conflict<1.0, but you have conflict 1.0.']
    assert_contains_expected_lines(result.stdout, expected_lines)