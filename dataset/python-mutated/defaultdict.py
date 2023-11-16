"""
This module demonstrates the use of defaultdict, which is a dictionary that is
possible to setup a default value in its creation.
"""
from collections import defaultdict
_GPA_MIN = 0.0
_GPA_MAX = 4.0
_EPS = 1e-06

def main():
    if False:
        for i in range(10):
            print('nop')
    student_gpa = defaultdict(float, [('john', 3.5), ('bob', 2.8), ('mary', 3.2)])
    assert len(student_gpa) == 3
    assert len(student_gpa.keys()) == len(student_gpa.values())
    student_names = []
    for student in student_gpa.keys():
        student_names.append(student)
    assert student_names == ['john', 'bob', 'mary']
    assert abs(student_gpa['john'] < 3.5) < _EPS
    assert student_gpa['jane'] == _GPA_MIN
    assert len(student_gpa) == 4

    def set_default_to_gpa_max():
        if False:
            for i in range(10):
                print('nop')
        return _GPA_MAX
    student_gpa.default_factory = set_default_to_gpa_max
    assert student_gpa['rika'] == _GPA_MAX
    assert len(student_gpa) == 5
if __name__ == '__main__':
    main()