from numpy.distutils.from_template import process_str
from numpy.testing import assert_equal
pyf_src = '\npython module foo\n    <_rd=real,double precision>\n    interface\n        subroutine <s,d>foosub(tol)\n            <_rd>, intent(in,out) :: tol\n        end subroutine <s,d>foosub\n    end interface\nend python module foo\n'
expected_pyf = '\npython module foo\n    interface\n        subroutine sfoosub(tol)\n            real, intent(in,out) :: tol\n        end subroutine sfoosub\n        subroutine dfoosub(tol)\n            double precision, intent(in,out) :: tol\n        end subroutine dfoosub\n    end interface\nend python module foo\n'

def normalize_whitespace(s):
    if False:
        i = 10
        return i + 15
    '\n    Remove leading and trailing whitespace, and convert internal\n    stretches of whitespace to a single space.\n    '
    return ' '.join(s.split())

def test_from_template():
    if False:
        return 10
    'Regression test for gh-10712.'
    pyf = process_str(pyf_src)
    normalized_pyf = normalize_whitespace(pyf)
    normalized_expected_pyf = normalize_whitespace(expected_pyf)
    assert_equal(normalized_pyf, normalized_expected_pyf)