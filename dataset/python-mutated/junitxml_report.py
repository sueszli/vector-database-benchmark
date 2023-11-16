"""Parses and extracts data from JUnitXML format files.

Example usage, comparing nosetests and pytest test collection:
$ cd sdks/python
$ rm *.xml
$ tox --recreate -e py38-gcp
$ tox --recreate -e py38-gcp-pytest
$ python3 ../../.test-infra/junitxml_report.py nosetests*.xml | sort -u > nosetests.out
$ python3 ../../.test-infra/junitxml_report.py pytest*.xml | sort -u > pytest.out
$ diff -u nosetests.out pytest.out | less
"""
import sys
import xml.etree.ElementTree as et

def print_testsuite(testsuite):
    if False:
        i = 10
        return i + 15
    assert testsuite.tag == 'testsuite'
    for testcase in testsuite:
        assert testcase.tag == 'testcase'
        attrib = testcase.attrib
        status = ''
        for child in testcase:
            if child.tag == 'skipped':
                assert status == ''
                status = 'S'
            elif child.tag == 'failure':
                assert status == ''
                status = 'F'
            elif child.tag in ['system-err', 'system-out']:
                pass
            else:
                raise NotImplementedError('tag not supported: %s' % child.tag)
        print('%s.%s %s' % (attrib['classname'], attrib['name'], status))

def process_xml(filename):
    if False:
        i = 10
        return i + 15
    tree = et.parse(filename)
    root = tree.getroot()
    if root.tag == 'testsuites':
        for testsuite in root:
            print_testsuite(testsuite)
    else:
        print_testsuite(root)

def main():
    if False:
        print('Hello World!')
    for filename in sys.argv[1:]:
        process_xml(filename)
if __name__ == '__main__':
    main()