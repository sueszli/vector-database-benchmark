import re
import batch_operation_sample

def test_batch_operation_sample(capsys):
    if False:
        return 10
    batch_operation_sample.run_sample()
    (out, _) = capsys.readouterr()
    expected = '.*Company generated:.*Company created:.*.*Job created:.*Job created:.*.*Job updated:.*Engineer in Mountain View.*Job updated:.*Engineer in Mountain View.*.*Job deleted.*Job deleted.*.*Company deleted.*'
    assert re.search(expected, out, re.DOTALL)