import textwrap
import pytest
import pytest_bdd as bdd
bdd.scenarios('hints.feature')

@pytest.fixture(autouse=True)
def set_up_word_hints(tmpdir, quteproc):
    if False:
        for i in range(10):
            print('nop')
    dict_file = tmpdir / 'dict'
    dict_file.write(textwrap.dedent('\n        one\n        two\n        three\n        four\n        five\n        six\n        seven\n        eight\n        nine\n        ten\n        eleven\n        twelve\n        thirteen\n    '))
    quteproc.set_setting('hints.dictionary', str(dict_file))