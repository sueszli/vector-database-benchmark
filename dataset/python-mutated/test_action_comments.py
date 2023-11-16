"""Tests for isort action comments, such as isort: skip"""
import isort

def test_isort_off_and_on():
    if False:
        while True:
            i = 10
    'Test so ensure isort: off action comment and associated on action comment work together'
    assert isort.code('# isort: off\nimport a\nimport a\n\n# isort: on\nimport a\nimport a\n') == '# isort: off\nimport a\nimport a\n\n# isort: on\nimport a\n'
    assert isort.code('\nimport a\nimport a\n\n# isort: off\nimport a\nimport a\n') == '\nimport a\n\n# isort: off\nimport a\nimport a\n'