from pathlib import Path
import pytest
from hypothesistooling.__main__ import PYTHONS
ci_checks = '    '.join((line.strip() for line in Path('.github/workflows/main.yml').read_text(encoding='utf-8').splitlines() if '- check-py' in line))

@pytest.mark.parametrize('version', sorted(PYTHONS))
def test_python_versions_are_tested_in_ci(version):
    if False:
        i = 10
        return i + 15
    slug = version.replace('pypy', 'py').replace('.', '')
    assert f'- check-py{slug}' in ci_checks, f'Add {version} to main.yml and tox.ini'