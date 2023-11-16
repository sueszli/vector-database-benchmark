import sys
import os
from os import listdir
from os.path import isfile, join
import time
import json
import datetime
from pytz import timezone
from dateutil import parser
'\nThis script will summary failed tests results and determine if any of them may be intermittents.  For\ntests that are determined to be intermittents, a dictionary structure will be generated to store information\nabout the intermittents.\n\nCurrently, a simple threshold test is used to determine if a test is intermittent.  If the failure count of\nany test exceed the threshold, we will label it as intermittent.  In particular, the following information\nwill be stored for each intermittent failure:\n        "JenkinsJobName"\n        "BuildID"\n        "Timestamp"\n        "GitHash"\n        "TestCategory"\n        "NodeName"\n        "FailureMessages"\n        "FailureCount"\n\n'
g_test_root_dir = os.path.dirname(os.path.realpath(__file__))
g_threshold_failure = 0
g_summary_dict_name = ''
g_summary_csv_filename = ''
g_file_start = []
g_summary_dict_intermittents = dict()
g_summary_dict_all = dict()

def init_intermittents_dict(init_dict):
    if False:
        for i in range(10):
            print('nop')
    '\n    initialize the fields of dictionary storing failed tests.\n    :return:\n    '
    init_dict['TestName'] = []
    init_dict['TestInfo'] = []

def usage():
    if False:
        i = 10
        return i + 15
    '\n    Print USAGE help.\n    '
    print('')
    print('Usage:  ')
    print('python summarizeINtermittents threshold Filename_for_dict AWS_path Failed_PyUnits_summary_dict_from ....')
    print('- threshold is an integer for which a failed test is labeled intermittent if its number of failure exceeds it.')
    print('- Filename_for_dict is a string denoting the name of the dictionary that will store the final intermittents.')
    print('- Failed_PyUnits_summary_dict_from is a string denoting the beginning of pickle files that contains')
    print('- ... denotes extra strings that represent the beginning of pickle files that you want us to summarizefor you.')

def summarizeFailedRuns():
    if False:
        return 10
    '\n    This function will look at the local directory and pick out files that have the correct start name and\n    summarize the results into one giant dict.\n\n    :return: None\n    '
    global g_summary_dict_all
    onlyFiles = [x for x in listdir(g_test_root_dir) if isfile(join(g_test_root_dir, x))]
    for f in onlyFiles:
        for fileStart in g_file_start:
            if fileStart in f and os.path.getsize(f) > 10:
                fFullPath = os.path.join(g_test_root_dir, f)
                try:
                    temp_dict = json.load(open(fFullPath, 'r'))
                    for ind in range(len(temp_dict['TestName'])):
                        addFailedTests(g_summary_dict_all, temp_dict, ind)
                except:
                    continue
                break

def addFailedTests(summary_dict, temp_dict, index):
    if False:
        return 10
    testName = temp_dict['TestName'][index]
    testNameList = summary_dict['TestName']
    if testName in testNameList:
        testIndex = testNameList.index(testName)
        updateFailedTestInfo(summary_dict, temp_dict['TestInfo'][index], testIndex, testName, False)
    else:
        summary_dict['TestName'].append(testName)
        updateFailedTestInfo(summary_dict, temp_dict['TestInfo'][index], len(summary_dict['TestName']) - 1, testName, True)

def updateFailedTestInfo(summary_dict, one_test_info, testIndex, testName, newTest):
    if False:
        while True:
            i = 10
    '\n    For each test, a dictionary structure will be built to record the various info about that test\'s failure\n    information.  In particular, for each failed tests, there will be a dictionary associated with that test\n    stored in the field "TestInfo" of g_faiiled_tests_info_dict.  The following fields are included:\n        "JenkinsJobName": job name\n        "BuildID"\n        "Timestamp": in seconds\n        "GitHash"\n        "TestCategory": JUnit, PyUnit, RUnit or HadoopPyUnit, HadoopRUnit\n        "NodeName": name of machine that the job was run on\n        "FailureCount": integer counting number of times this particular test has failed.  An intermittent can be\n          determined as any test with FailureCount >= 2.\n        "FailureMessages": contains failure messages for the test\n    :return: a new dict for that test\n    '
    if newTest:
        summary_dict['TestInfo'].append(dict())
        summary_dict['TestInfo'][testIndex]['JenkinsJobName'] = []
        summary_dict['TestInfo'][testIndex]['BuildID'] = []
        summary_dict['TestInfo'][testIndex]['Timestamp'] = []
        summary_dict['TestInfo'][testIndex]['GitHash'] = []
        summary_dict['TestInfo'][testIndex]['TestCategory'] = []
        summary_dict['TestInfo'][testIndex]['NodeName'] = []
        summary_dict['TestInfo'][testIndex]['FailureCount'] = 0
        summary_dict['TestInfo'][testIndex]['TestName'] = testName
        summary_dict['TestInfo'][testIndex]['FailureMessages'] = []
    summary_dict['TestInfo'][testIndex]['JenkinsJobName'].extend(one_test_info['JenkinsJobName'])
    summary_dict['TestInfo'][testIndex]['BuildID'].extend(one_test_info['BuildID'])
    summary_dict['TestInfo'][testIndex]['Timestamp'].extend(one_test_info['Timestamp'])
    summary_dict['TestInfo'][testIndex]['GitHash'].extend(one_test_info['GitHash'])
    summary_dict['TestInfo'][testIndex]['TestCategory'].extend(one_test_info['TestCategory'])
    summary_dict['TestInfo'][testIndex]['NodeName'].extend(one_test_info['NodeName'])
    summary_dict['TestInfo'][testIndex]['FailureMessages'].extend(one_test_info['FailureMessages'])
    summary_dict['TestInfo'][testIndex]['FailureCount'] += one_test_info['FailureCount']

def extractPrintSaveIntermittens():
    if False:
        print('Hello World!')
    '\n    This function will print out the intermittents onto the screen for casual viewing.  It will also print out\n    where the giant summary dictionary is going to be stored.\n\n    :return: None\n    '
    global g_summary_dict_intermittents
    localtz = time.tzname[0]
    for ind in range(len(g_summary_dict_all['TestName'])):
        if g_summary_dict_all['TestInfo'][ind]['FailureCount'] >= g_threshold_failure:
            addFailedTests(g_summary_dict_intermittents, g_summary_dict_all, ind)
    if len(g_summary_dict_intermittents['TestName']) > 0:
        json.dump(g_summary_dict_intermittents, open(g_summary_dict_name, 'w'))
        with open(g_summary_csv_filename, 'w') as summaryFile:
            for ind in range(len(g_summary_dict_intermittents['TestName'])):
                testName = g_summary_dict_intermittents['TestName'][ind]
                numberFailure = g_summary_dict_intermittents['TestInfo'][ind]['FailureCount']
                firstFailedTS = parser.parse(time.ctime(min(g_summary_dict_intermittents['TestInfo'][ind]['Timestamp'])) + ' ' + localtz)
                firstFailedStr = firstFailedTS.strftime('%a %b %d %H:%M:%S %Y %Z')
                recentFail = parser.parse(time.ctime(max(g_summary_dict_intermittents['TestInfo'][ind]['Timestamp'])) + ' ' + localtz)
                recentFailStr = recentFail.strftime('%a %b %d %H:%M:%S %Y %Z')
                eachTest = '{0}, {1}, {2}, {3}\n'.format(testName, recentFailStr, numberFailure, g_summary_dict_intermittents['TestInfo'][ind]['TestCategory'][0])
                summaryFile.write(eachTest)
                print('Intermittent: {0}, Last failed: {1}, Failed {2} times since {3}'.format(testName, recentFailStr, numberFailure, firstFailedStr))

def main(argv):
    if False:
        while True:
            i = 10
    '\n    Main program.  Expect script name plus  inputs in the following order:\n    - This script name\n    1. threshold: integer that will denote when a failed test will be declared an intermittent\n    2. string denote filename of where our final dict structure will be stored.\n    3. string that denote the beginning of a file containing failed tests info.\n    4. Optional strings that denote the beginning of a file containing failed tests info.\n\n    @return: none\n    '
    global g_script_name
    global g_test_root_dir
    global g_threshold_failure
    global g_file_start
    global g_summary_dict_name
    global g_summary_dict_all
    global g_summary_dict_intermittents
    global g_summary_csv_filename
    if len(argv) < 5:
        print('Wrong call.  Not enough arguments.\n')
        usage()
        sys.exit(1)
    else:
        g_threshold_failure = int(argv[1])
        g_summary_dict_name = os.path.join(g_test_root_dir, argv[2])
        g_summary_csv_filename = g_summary_dict_name + '.csv'
        for ind in range(3, len(argv)):
            g_file_start.append(argv[ind])
        init_intermittents_dict(g_summary_dict_all)
        init_intermittents_dict(g_summary_dict_intermittents)
        summarizeFailedRuns()
        extractPrintSaveIntermittens()
if __name__ == '__main__':
    main(sys.argv)