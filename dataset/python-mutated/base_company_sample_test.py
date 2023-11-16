import re
import base_company_sample

def test_base_company_sample(capsys):
    if False:
        return 10
    base_company_sample.run_sample()
    (out, _) = capsys.readouterr()
    expected = '.*Company generated:.*\n.*Company created:.*\n.*Company existed:.*\n.*Company updated:.*elgoog.*\n.*Company updated:.*changedTitle.*\n.*Company deleted.*\n'
    assert re.search(expected, out, re.DOTALL)