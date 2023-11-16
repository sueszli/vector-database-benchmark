import os
import re
import pytest
import translate_from_file
RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

@pytest.mark.skip(reason='skip test for deprecated code sample')
def test_translate_streaming(capsys):
    if False:
        for i in range(10):
            print('nop')
    translate_from_file.translate_from_file(os.path.join(RESOURCES, 'audio.raw'))
    (out, err) = capsys.readouterr()
    assert re.search('Partial translation', out, re.DOTALL | re.I)