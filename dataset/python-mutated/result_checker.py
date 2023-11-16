"""
This script runs as part of the Travis CI build on Github and controls
whether a patch passes the regression test suite.

Given the failures encountered by the regression suite runner (run.py) in
  ../results/whitelist.csv
and the current whitelist of failures considered acceptable in
  ./run_regression_suite_failures_whitelisted.csv
determine PASSED or FAILED.

"""
import sys
import os
RESULTS_FILE = os.path.join('..', 'results', 'run_regression_suite_failures.csv')
WHITELIST_FILE = os.path.join('whitelist.csv')
BANNER = '\n*****************************************************************\nRegression suite result checker\n(test/regression/result_checker.py)\n*****************************************************************\n'

def passed(message):
    if False:
        print('Hello World!')
    print('\n\n**PASSED: {0}.\n'.format(message))
    return 0

def failed(message):
    if False:
        i = 10
        return i + 15
    print('\n\n**FAILED: {0}. \nFor more information see test/regression/README.\n'.format(message))
    return -1

def read_results_csv(filename):
    if False:
        while True:
            i = 10
    parse = lambda line: map(str.strip, line.split(';')[:2])
    try:
        with open(filename, 'rt') as results:
            return dict((parse(line) for line in results.readlines()[1:]))
    except IOError:
        print('Failed to read {0}.'.format(filename))
        return None

def run():
    if False:
        while True:
            i = 10
    print(BANNER)
    print('Reading input files.')
    result_dict = read_results_csv(RESULTS_FILE)
    whitelist_dict = read_results_csv(WHITELIST_FILE)
    if result_dict is None or whitelist_dict is None:
        return failed('Could not locate input files')
    if not result_dict:
        return passed('No failures encountered')
    print('Failures:\n' + '\n'.join(sorted(result_dict.keys())))
    print('Whitelisted:\n' + '\n'.join(sorted(whitelist_dict.keys())))
    non_whitelisted_failures = set(result_dict.keys()) - set(whitelist_dict.keys())
    print('Failures not whitelisted:\n' + '\n'.join(sorted(non_whitelisted_failures)))
    if not non_whitelisted_failures:
        return passed('All failures are whitelisted and considered acceptable \n' + 'due to implementation differences, library shortcomings and bugs \n' + 'that have not been fixed for a long time')
    return failed('Encountered new regression failures that are not whitelisted.  \n' + 'Please carefully review the changes you made and use the gen_db.py script\n' + 'to update the regression database for the affected files')
if __name__ == '__main__':
    sys.exit(run())