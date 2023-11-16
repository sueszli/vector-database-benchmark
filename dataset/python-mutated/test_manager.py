import itertools
import pytest
from redbot.cogs.audio.manager import LavalinkOldVersion, LavalinkVersion
ORDERED_VERSIONS = [LavalinkOldVersion('3.3.2.3', build_number=1239), LavalinkOldVersion('3.4.0', build_number=1275), LavalinkOldVersion('3.4.0', build_number=1350), LavalinkVersion(3, 3), LavalinkVersion(3, 4), LavalinkVersion(3, 5, rc=1), LavalinkVersion(3, 5, rc=2), LavalinkVersion(3, 5, rc=3), LavalinkVersion(3, 5, rc=3, red=1), LavalinkVersion(3, 5, rc=3, red=2), LavalinkVersion(3, 5), LavalinkVersion(3, 5, red=1), LavalinkVersion(3, 5, red=2), LavalinkVersion(3, 5, 1)]

@pytest.mark.parametrize('raw_version,raw_build_number,expected', (('3.4.0', '1350', LavalinkOldVersion('3.4.0', build_number=1350)), ('3.3.2.3', '1239', LavalinkOldVersion('3.3.2.3', build_number=1239)), ('3.3.1', '987', LavalinkOldVersion('3.3.1', build_number=987))))
def test_old_ll_version_parsing(raw_version: str, raw_build_number: str, expected: LavalinkOldVersion) -> None:
    if False:
        for i in range(10):
            print('nop')
    line = b'Version: %b\nBuild: %b' % (raw_version.encode(), raw_build_number.encode())
    assert LavalinkOldVersion.from_version_output(line)

@pytest.mark.parametrize('raw_version,expected', (('3.5-rc4', LavalinkVersion(3, 5, rc=4)), ('3.5', LavalinkVersion(3, 5)), ('3.6.0-rc.1', LavalinkVersion(3, 6, 0, rc=1)), ('3.7.5-rc.1+red.1', LavalinkVersion(3, 7, 5, rc=1, red=1)), ('3.7.5-rc.1+red.123', LavalinkVersion(3, 7, 5, rc=1, red=123)), ('3.7.5', LavalinkVersion(3, 7, 5)), ('3.7.5+red.1', LavalinkVersion(3, 7, 5, red=1)), ('3.7.5+red.123', LavalinkVersion(3, 7, 5, red=123))))
def test_ll_version_parsing(raw_version: str, expected: LavalinkVersion) -> None:
    if False:
        for i in range(10):
            print('nop')
    line = b'Version: ' + raw_version.encode()
    assert LavalinkVersion.from_version_output(line)

def test_ll_version_comparison() -> None:
    if False:
        print('Hello World!')
    (it1, it2) = itertools.tee(ORDERED_VERSIONS)
    next(it2, None)
    for (a, b) in zip(it1, it2):
        assert a < b