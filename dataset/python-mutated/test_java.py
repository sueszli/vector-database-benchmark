import pytest
from symbolic.proguard import ProguardMapper
from sentry.profiles.java import deobfuscate_signature
PROGUARD_SOURCE = b'# compiler: R8\n# compiler_version: 2.0.74\n# min_api: 16\n# pg_map_id: 5b46fdc\n# common_typos_disable\n# {"id":"com.android.tools.r8.mapping","version":"1.0"}\norg.slf4j.helpers.Util$ClassContextSecurityManager -> org.a.b.g$a:\n    65:65:void <init>() -> <init>\n    67:67:java.lang.Class[] getClassContext() -> a\n    69:69:java.lang.Class[] getExtraClassContext() -> a\n    65:65:void <init>(org.slf4j.helpers.Util$1) -> <init>\n'

@pytest.fixture
def mapper(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    mapping_file_path = str(tmp_path.joinpath('mapping_file'))
    with open(mapping_file_path, 'wb') as f:
        f.write(PROGUARD_SOURCE)
    mapper = ProguardMapper.open(mapping_file_path)
    assert mapper.has_line_info
    return mapper

@pytest.mark.parametrize(['obfuscated', 'expected'], [('', ''), ('()', ''), ('(L)', ''), ('()V', '()'), ('([I)V', '(int[])'), ('(III)V', '(int, int, int)'), ('([Ljava/lang/String;)V', '(java.lang.String[])'), ('([[J)V', '(long[][])'), ('(I)I', '(int): int'), ('([B)V', '(byte[])'), ('(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;', '(java.lang.String, java.lang.String): java.lang.String')])
def test_deobfuscate_signature(mapper, obfuscated, expected):
    if False:
        while True:
            i = 10
    assert deobfuscate_signature(obfuscated, mapper) == expected