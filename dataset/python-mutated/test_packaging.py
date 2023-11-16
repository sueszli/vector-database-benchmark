from typing import Optional, Tuple
import pytest
from pip._vendor.packaging import specifiers
from pip._vendor.packaging.requirements import Requirement
from pip._internal.utils.packaging import check_requires_python, get_requirement

@pytest.mark.parametrize('version_info, requires_python, expected', [((3, 6, 5), '== 3.6.4', False), ((3, 6, 5), '== 3.6.5', True), ((3, 6, 5), None, True)])
def test_check_requires_python(version_info: Tuple[int, int, int], requires_python: Optional[str], expected: bool) -> None:
    if False:
        print('Hello World!')
    actual = check_requires_python(requires_python, version_info)
    assert actual == expected

def test_check_requires_python__invalid() -> None:
    if False:
        print('Hello World!')
    '\n    Test an invalid Requires-Python value.\n    '
    with pytest.raises(specifiers.InvalidSpecifier):
        check_requires_python('invalid', (3, 6, 5))

def test_get_or_create_caching() -> None:
    if False:
        print('Hello World!')
    'test caching of get_or_create requirement'
    teststr = 'affinegap==1.10'
    from_helper = get_requirement(teststr)
    freshly_made = Requirement(teststr)
    for iattr in ['name', 'url', 'extras', 'specifier', 'marker']:
        assert getattr(from_helper, iattr) == getattr(freshly_made, iattr)
    assert get_requirement(teststr) is not Requirement(teststr)
    assert get_requirement(teststr) is get_requirement(teststr)