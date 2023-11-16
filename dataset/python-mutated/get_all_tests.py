"""
Gets all tests cases from xunit file.
"""
from __future__ import annotations
import sys
from xml.etree import ElementTree

def last_replace(s, old, new, number_of_occurrences):
    if False:
        while True:
            i = 10
    '\n    Replaces last n occurrences of the old string with the new one within the string provided\n\n    :param s: string to replace occurrences with\n    :param old: old string\n    :param new: new string\n    :param number_of_occurrences: how many occurrences should be replaced\n    :return: string with last n occurrences replaced\n    '
    list_of_components = s.rsplit(old, number_of_occurrences)
    return new.join(list_of_components)

def print_all_cases(xunit_test_file_path):
    if False:
        return 10
    '\n    Prints all test cases read from the xunit test file\n\n    :param xunit_test_file_path: path of the xunit file\n    :return: None\n    '
    with open(xunit_test_file_path) as file:
        text = file.read()
    root = ElementTree.fromstring(text)
    test_cases = root.findall('.//testcase')
    classes = set()
    modules = set()
    for test_case in test_cases:
        the_module = test_case['classname'].rpartition('.')[0]
        the_class = last_replace(test_case.get('classname'), '.', ':', 1)
        test_method = test_case.get('name')
        modules.add(the_module)
        classes.add(the_class)
        print(the_class + '.' + test_method)
    for the_class in classes:
        print(the_class)
    for the_module in modules:
        print(the_module)
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please provide name of xml unit file as first parameter')
        sys.exit(1)
    file_name = sys.argv[1]
    print_all_cases(file_name)