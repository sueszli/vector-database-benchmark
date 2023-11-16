"""Tests for the minimum dependencies in the README.rst file."""
import os
import platform
import re
from pathlib import Path
import pytest
from sklearn.utils.fixes import parse_version
import imblearn
from imblearn._min_dependencies import dependent_packages

@pytest.mark.skipif(platform.system() == 'Windows', reason='This test is enough on unix system')
def test_min_dependencies_readme():
    if False:
        for i in range(10):
            print('nop')
    pattern = re.compile('(\\.\\. \\|)' + '(([A-Za-z]+\\-?)+)' + '(MinVersion\\| replace::)' + '( [0-9]+\\.[0-9]+(\\.[0-9]+)?)')
    readme_path = Path(imblearn.__path__[0]).parents[0]
    readme_file = readme_path / 'README.rst'
    if not os.path.exists(readme_file):
        pytest.skip('The README.rst file is not available.')
    with readme_file.open('r') as f:
        for line in f:
            matched = pattern.match(line)
            if not matched:
                continue
            (package, version) = (matched.group(2), matched.group(5))
            package = package.lower()
            if package in dependent_packages:
                version = parse_version(version)
                min_version = parse_version(dependent_packages[package][0])
                assert version == min_version, f'{package} has a mismatched version'