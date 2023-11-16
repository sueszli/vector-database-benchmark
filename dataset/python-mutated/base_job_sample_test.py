import re
import base_job_sample

def test_base_job_sample(capsys):
    if False:
        print('Hello World!')
    base_job_sample.run_sample()
    (out, _) = capsys.readouterr()
    expected = '.*Job generated:.*\n.*Job created:.*\n.*Job existed:.*\n.*Job updated:.*changedDescription.*\n.*Job updated:.*changedJobTitle.*\n.*Job deleted.*\n'
    assert re.search(expected, out)