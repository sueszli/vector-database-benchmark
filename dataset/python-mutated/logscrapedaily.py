import sys
import os
import json
import pickle
import copy
import subprocess
'\nIn summary, this script given a jenkins job full console url and a summary log filename will\n1. scrape the console output log, all unit tests outputs and all java_*_0.out.txt of the\n   latest build.\n2. From all the logs, it will generate potentially two log files: jenkins_job_name_build_number_failed_tests.log\n   and jenkins_job_name_build_number_passed_tests.log.  Inside each log file, it contains the job name, build number,\n   timestamp, git hash, git branch, node name, build failure and build timeout information.  In addition, it will list\n   unit tests that failed/passed with the corresponding java WARN/ERRR/FATAL/STACKTRACE messages associated with the unit tests.\n3. Users can choose to ignore certain java messages that are deemed okay.  These ignored java messages are stored in a pickle\n   file with a default name and location.  However, if the user wants to use their own ignored java messages, they can do\n   so by specifying a third optional argument to this script as the name to where their own personal pickle file name.\n4. If there are okay ignored java messages stored in a pickle file, this script will not grab them and store them in\n   any log files.\n5. For details on how to generate ignore java messages and save them to a pickle file, please see addjavamessage2ignore.py.\n'
g_test_root_dir = os.path.dirname(os.path.realpath(__file__))
g_script_name = ''
g_node_name = 'Building remotely on'
g_git_hash_branch = 'Checking out Revision'
g_build_timeout = 'Build timed out'
g_build_success = ['Finished: SUCCESS', 'BUILD SUCCESSFUL']
g_build_success_tests = ['generate_rest_api_docs.py', 'generate_java_bindings.py']
g_build_id_text = 'Build id is'
g_view_name = ''
g_temp_filename = os.path.join(g_test_root_dir, 'tempText')
g_output_filename_failed_tests = os.path.join(g_test_root_dir, 'failedMessage_failed_tests.log')
g_output_filename_passed_tests = os.path.join(g_test_root_dir, 'failedMessage_passed_tests.log')
g_output_pickle_filename = os.path.join(g_test_root_dir, 'failedMessage.pickle.log')
g_failed_test_info_dict = {}
g_failed_test_info_dict['7.build_failure'] = 'No'
g_weekdays = 'Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday'
g_months = 'January, Feburary, March, May, April, May, June, July, August, September, October, November, December'
g_failure_occurred = False
g_failed_jobs = []
g_failed_job_java_message_types = []
g_failed_job_java_messages = []
g_success_jobs = []
g_success_job_java_message_types = []
g_success_job_java_messages = []
g_before_java_file = ['H2O Cloud', 'Node', 'started with output file']
g_java_filenames = []
g_java_message_type = ['WARN:', ':WARN:', 'ERRR:', 'FATAL:', 'TRACE:']
g_all_java_message_type = ['WARN:', ':WARN:', 'ERRR:', 'FATAL:', 'TRACE:', 'DEBUG:', 'INFO:']
g_java_general_bad_message_types = []
g_java_general_bad_messages = []
g_jenkins_url = ''
g_toContinue = False
g_current_testname = ''
g_java_start_text = 'STARTING TEST:'
g_ok_java_messages = {}
g_java_message_pickle_filename = 'bad_java_messages_to_exclude.pickle'
g_build_failed_message = ['Finished: FAILURE'.lower(), 'BUILD FAILED'.lower()]
g_summary_text_filename = ''
'\nThe sole purpose of this function is to enable us to be able to call\nany function that is specified as the first argument using the argument\nlist specified in second argument.\n'

def perform(function_name, *arguments):
    if False:
        return 10
    '\n\n    Parameters\n    ----------\n\n    function_name :  python function handle\n        name of functio we want to call and run\n    *arguments :  Python list\n        list of arguments to be passed to function_name\n\n\n    :return: bool\n    '
    return function_name(*arguments)
'\nThis function is written to remove extra characters before the actual string we are\nlooking for.  The Jenkins console output is encoded using utf-8.  However, the stupid\nredirect function can only encode using ASCII.  I have googled for half a day with no\nresults to how.  Hence, we are going to the heat and just manually get rid of the junk.\n'

def extract_true_string(string_content):
    if False:
        return 10
    "\n    remove extra characters before the actual string we are\n    looking for.  The Jenkins console output is encoded using utf-8.  However, the stupid\n    redirect function can only encode using ASCII.  I have googled for half a day with no\n    results to how to resolve the issue.  Hence, we are going to the heat and just manually\n    get rid of the junk.\n\n    Parameters\n    ----------\n\n    string_content :  str\n        contains a line read in from jenkins console\n\n    :return: str: contains the content of the line after the string '[0m'\n\n    "
    (startL, found, endL) = string_content.partition('[0m')
    if found:
        return endL
    else:
        return string_content
'\nFunction find_time is written to extract the timestamp when a job is built.\n'

def find_time(each_line, temp_func_list):
    if False:
        print('Hello World!')
    '\n    calculate the approximate date/time from the timestamp about when the job\n    was built.  This information was then saved in dict g_failed_test_info_dict.\n    In addition, it will delete this particular function handle off the temp_func_list\n    as we do not need to perform this action again.\n\n    Parameters\n    ----------\n\n    each_line :  str\n        contains a line read in from jenkins console\n    temp_func_list :  list of Python function handles\n        contains a list of functions that we want to invoke to extract information from\n        the Jenkins console text.\n\n    :return: bool to determine if text mining should continue on the jenkins console text\n    '
    global g_weekdays
    global g_months
    global g_failed_test_info_dict
    temp_strings = each_line.strip().split()
    if len(temp_strings) > 2:
        if (temp_strings[0] in g_weekdays or temp_strings[1] in g_weekdays) and (temp_strings[1] in g_months or temp_strings[2] in g_months):
            g_failed_test_info_dict['3.timestamp'] = each_line.strip()
            temp_func_list.remove(find_time)
    return True

def find_node_name(each_line, temp_func_list):
    if False:
        print('Hello World!')
    '\n    Find the slave machine where a Jenkins job was executed on.  It will save this\n    information in g_failed_test_info_dict.  In addition, it will\n    delete this particular function handle off the temp_func_list as we do not need\n    to perform this action again.\n\n    Parameters\n    ----------\n\n    each_line :  str\n        contains a line read in from jenkins console\n    temp_func_list :  list of Python function handles\n        contains a list of functions that we want to invoke to extract information from\n        the Jenkins console text.\n\n    :return: bool to determine if text mining should continue on the jenkins console text\n    '
    global g_node_name
    global g_failed_test_info_dict
    if g_node_name in each_line:
        temp_strings = each_line.split()
        [start, found, endstr] = each_line.partition(g_node_name)
        if found:
            temp_strings = endstr.split()
            g_failed_test_info_dict['6.node_name'] = extract_true_string(temp_strings[1])
            temp_func_list.remove(find_node_name)
    return True

def find_git_hash_branch(each_line, temp_func_list):
    if False:
        i = 10
        return i + 15
    '\n    Find the git hash and branch info that  a Jenkins job was taken from.  It will save this\n    information in g_failed_test_info_dict.  In addition, it will delete this particular\n    function handle off the temp_func_list as we do not need to perform this action again.\n\n    Parameters\n    ----------\n\n    each_line :  str\n        contains a line read in from jenkins console\n    temp_func_list :  list of Python function handles\n        contains a list of functions that we want to invoke to extract information from\n        the Jenkins console text.\n\n    :return: bool to determine if text mining should continue on the jenkins console text\n    '
    global g_git_hash_branch
    global g_failed_test_info_dict
    if g_git_hash_branch in each_line:
        [start, found, endstr] = each_line.partition(g_git_hash_branch)
        temp_strings = endstr.strip().split()
        if len(temp_strings) > 1:
            g_failed_test_info_dict['4.git_hash'] = temp_strings[0]
            g_failed_test_info_dict['5.git_branch'] = temp_strings[1]
        temp_func_list.remove(find_git_hash_branch)
    return True

def find_build_timeout(each_line, temp_func_list):
    if False:
        print('Hello World!')
    '\n    Find if a Jenkins job has taken too long to finish and was killed.  It will save this\n    information in g_failed_test_info_dict.\n\n    Parameters\n    ----------\n\n    each_line :  str\n        contains a line read in from jenkins console\n    temp_func_list :  list of Python function handles\n        contains a list of functions that we want to invoke to extract information from\n        the Jenkins console text.\n\n    :return: bool to determine if text mining should continue on the jenkins console text\n'
    global g_build_timeout
    global g_failed_test_info_dict
    global g_failure_occurred
    if g_build_timeout in each_line:
        g_failed_test_info_dict['8.build_timeout'] = 'Yes'
        g_failure_occurred = True
        return False
    else:
        return True

def find_build_failure(each_line, temp_func_list):
    if False:
        while True:
            i = 10
    '\n    Find if a Jenkins job has failed to build.  It will save this\n    information in g_failed_test_info_dict.  In addition, it will delete this particular\n    function handle off the temp_func_list as we do not need to perform this action again.\n\n    Parameters\n    ----------\n\n    each_line :  str\n        contains a line read in from jenkins console\n    temp_func_list :  list of Python function handles\n        contains a list of functions that we want to invoke to extract information from\n        the Jenkins console text.\n\n    :return: bool to determine if text mining should continue on the jenkins console text\n    '
    global g_build_success
    global g_build_success_tests
    global g_failed_test_info_dict
    global g_failure_occurred
    global g_build_failed_message
    for ind in range(0, len(g_build_failed_message)):
        if g_build_failed_message[ind] in each_line.lower():
            if ind == 0 and len(g_failed_jobs) > 0:
                continue
            else:
                g_failure_occurred = True
                g_failed_test_info_dict['7.build_failure'] = 'Yes'
                temp_func_list.remove(find_build_failure)
                return False
    return True

def find_java_filename(each_line, temp_func_list):
    if False:
        return 10
    '\n    Find if all the java_*_0.out.txt files that were mentioned in the console output.\n    It will save this information in g_java_filenames as a list of strings.\n\n    Parameters\n    ----------\n\n    each_line :  str\n        contains a line read in from jenkins console\n    temp_func_list :  list of Python function handles\n        contains a list of functions that we want to invoke to extract information from\n        the Jenkins console text.\n\n    :return: bool to determine if text mining should continue on the jenkins console text\n'
    global g_before_java_file
    global g_java_filenames
    for each_word in g_before_java_file:
        if each_word not in each_line:
            return True
    temp_strings = each_line.split()
    g_java_filenames.append(temp_strings[-1])
    return True

def find_build_id(each_line, temp_func_list):
    if False:
        for i in range(10):
            print('nop')
    '\n    Find the build id of a jenkins job.  It will save this\n    information in g_failed_test_info_dict.  In addition, it will delete this particular\n    function handle off the temp_func_list as we do not need to perform this action again.\n\n    Parameters\n    ----------\n\n    each_line :  str\n        contains a line read in from jenkins console\n    temp_func_list :  list of Python function handles\n        contains a list of functions that we want to invoke to extract information from\n        the Jenkins console text.\n\n    :return: bool to determine if text mining should continue on the jenkins console text\n    '
    global g_before_java_file
    global g_java_filenames
    global g_build_id_text
    global g_jenkins_url
    global g_output_filename
    global g_output_pickle_filename
    if g_build_id_text in each_line:
        [startStr, found, endStr] = each_line.partition(g_build_id_text)
        g_failed_test_info_dict['2.build_id'] = endStr.strip()
        temp_func_list.remove(find_build_id)
        g_jenkins_url = os.path.join('http://', g_jenkins_url, 'view', g_view_name, 'job', g_failed_test_info_dict['1.jobName'], g_failed_test_info_dict['2.build_id'], 'artifact')
    return True
g_build_func_list = [find_time, find_node_name, find_build_id, find_git_hash_branch, find_build_timeout, find_build_failure, find_java_filename]

def update_test_dict(each_line):
    if False:
        while True:
            i = 10
    '\n    Extract unit tests information from the jenkins job console output.  It will save this\n    information in g_failed_jobs list and setup a place holder for saving the bad java\n    messages/message types in g_failed_job_java_messages, g_failed_job_java_message_types.\n\n    Parameters\n    ----------\n\n    each_line :  str\n        contains a line read in from jenkins console\n\n    :return: bool to determine if text mining should continue on the jenkins console text\n    '
    global g_ignore_test_names
    global g_failed_jobs
    global g_failed_job_java_messages
    global g_failure_occurred
    temp_strings = each_line.split()
    if len(temp_strings) >= 5 and 'FAIL' in each_line and ('FAILURE' not in each_line):
        test_name = temp_strings[-2]
        g_failed_jobs.append(test_name)
        g_failed_job_java_messages.append([])
        g_failed_job_java_message_types.append([])
        g_failure_occurred = True
    return True
'\nThis function is written to extract the error messages from console output and\npossible from the java_*_*.out to warn users of potentially bad runs.\n\n'

def extract_test_results():
    if False:
        while True:
            i = 10
    '\n    Extract error messages from jenkins console output and from java_*_0.out.txt if they exist to\n    warn users of potentially bad tests.  In addition, it will grab the following info about the jenkins\n    job from the console output and saved it into g_failed_test_info_dict:\n    1.jobName\n    2.build_id\n    3.timestamp\n    4.git_hash\n    5.git_branch\n    6.node_name\n    7.build_failure\n    8.build_timeout\n    9.general_bad_java_messages\n    failed_tests_info *********: list of failed tests and their associated bad java messages\n    passed_tests_info *********: list of passed tests and their associated bad java messages\n\n    This is achieved by calling various functions.\n\n    :return: none\n    '
    global g_test_root_dir
    global g_temp_filename
    global g_output_filename
    global g_build_func_list
    temp_func_list = copy.copy(g_build_func_list)
    if os.path.isfile(g_temp_filename):
        console_file = open(g_temp_filename, 'r')
        for each_line in console_file:
            each_line.strip()
            for each_function in temp_func_list:
                to_continue = perform(each_function, each_line, temp_func_list)
                if not to_continue:
                    break
            if not to_continue:
                break
            else:
                update_test_dict(each_line)
        console_file.close()
    else:
        print('Error: console output file ' + g_temp_filename + ' does not exist.')
        sys.exit(1)
'\nThis function is written to extract the console output that has already been stored\nin a text file in a remote place and saved it in a local directory that we have accessed\nto.  We want to be able to read in the local text file and proces it.\n'

def get_console_out(url_string):
    if False:
        print('Hello World!')
    '\n    Grab the console output from Jenkins and save the content into a temp file\n     (g_temp_filename).\n\n    Parameters\n    ----------\n    url_string :  str\n        contains information on the jenkins job whose console output we are interested in.\n\n    :return: none\n    '
    global g_temp_filename
    full_command = 'curl ' + url_string + ' > ' + g_temp_filename
    subprocess.call(full_command, shell=True)

def extract_job_build_url(url_string):
    if False:
        return 10
    '\n    From user input, grab the jenkins job name and saved it in g_failed_test_info_dict.\n    In addition, it will grab the jenkins url and the view name into g_jenkins_url, and\n    g_view_name.\n\n    Parameters\n    ----------\n    url_string :  str\n        contains information on the jenkins job whose console output we are interested in.\n\n    :return: none\n    '
    global g_failed_test_info_dict
    global g_jenkins_url
    global g_view_name
    tempString = url_string.strip('/').split('/')
    if len(tempString) < 6:
        print('Illegal URL resource address.\n')
        sys.exit(1)
    g_failed_test_info_dict['1.jobName'] = tempString[6]
    g_jenkins_url = tempString[2]
    g_view_name = tempString[4]

def grab_java_message():
    if False:
        print('Hello World!')
    'scan through the java output text and extract the bad java messages that may or may not happened when\n    unit tests are run.  It will not record any bad java messages that are stored in g_ok_java_messages.\n\n    :return: none\n    '
    global g_temp_filename
    global g_current_testname
    global g_java_start_text
    global g_ok_java_messages
    global g_java_general_bad_messages
    global g_java_general_bad_message_types
    global g_failure_occurred
    global g_java_message_type
    global g_all_java_message_type
    global g_toContinue
    java_messages = []
    java_message_types = []
    if os.path.isfile(g_temp_filename):
        java_file = open(g_temp_filename, 'r')
        g_toContinue = False
        tempMessage = ''
        messageType = ''
        for each_line in java_file:
            if g_java_start_text in each_line:
                (startStr, found, endStr) = each_line.partition(g_java_start_text)
                if len(found) > 0:
                    if len(g_current_testname) > 0:
                        associate_test_with_java(g_current_testname, java_messages, java_message_types)
                    g_current_testname = endStr.strip()
                    java_messages = []
                    java_message_types = []
            temp_strings = each_line.strip().split()
            if len(temp_strings) >= 6 and temp_strings[5] in g_all_java_message_type:
                if g_toContinue == True:
                    addJavaMessages(tempMessage, messageType, java_messages, java_message_types)
                    tempMessage = ''
                    messageType = ''
                g_toContinue = False
            elif g_toContinue:
                tempMessage += each_line
            if len(temp_strings) > 5 and temp_strings[5] in g_java_message_type:
                (startStr, found, endStr) = each_line.partition(temp_strings[5])
                if found and len(endStr.strip()) > 0:
                    tempMessage += endStr
                    messageType = temp_strings[5]
                    g_toContinue = True
        java_file.close()

def addJavaMessages(tempMessage, messageType, java_messages, java_message_types):
    if False:
        for i in range(10):
            print('nop')
    '\n    Insert Java messages into java_messages and java_message_types if they are associated\n    with a unit test or into g_java_general_bad_messages/g_java_general_bad_message_types\n    otherwise.\n\n    Parameters\n    ----------\n    tempMessage :  str\n        contains the bad java messages\n    messageType :  str\n        contains the bad java message type\n    java_messages : list of str\n        contains the bad java message list associated with a unit test\n    java_message_tuypes :  list of str\n        contains the bad java message type list associated with a unit test.\n\n    :return: none\n    '
    global g_current_testname
    global g_java_general_bad_messages
    global g_java_general_bad_message_types
    global g_failure_occurred
    tempMess = tempMessage.strip()
    if tempMess not in g_ok_java_messages['general']:
        if len(g_current_testname) == 0:
            g_java_general_bad_messages.append(tempMess)
            g_java_general_bad_message_types.append(messageType)
            g_failure_occurred = True
        else:
            write_test = False
            if g_current_testname in g_ok_java_messages.keys() and tempMess in g_ok_java_messages[g_current_testname]:
                write_test = False
            else:
                write_test = True
            if write_test:
                java_messages.append(tempMess)
                java_message_types.append(messageType)
                g_failure_occurred = True

def associate_test_with_java(testname, java_message, java_message_type):
    if False:
        while True:
            i = 10
    '\n    When a new unit test is started as indicated in the java_*_0.out.txt file,\n    update the data structures that are keeping track of unit tests being run and\n    bad java messages/messages types associated with each unit test.  Since a new\n    unit test is being started, save all the bad java messages associated with\n    the previous unit test and start a new set for the new unit test.\n\n    Parameters\n    ----------\n    testname :  str\n        previous unit test testname\n    java_message :  list of str\n        bad java messages associated with testname\n    java_message_type :  list of str\n        bad java message types associated with testname\n\n    :return :  none\n    '
    global g_failed_jobs
    global g_failed_job_java_messages
    global g_failed_job_java_message_types
    global g_success_jobs
    global g_success_job_java_messages
    global g_success_job_java_message_types
    if len(java_message) > 0:
        if testname in g_failed_jobs:
            message_index = g_failed_jobs.index(testname)
            g_failed_job_java_messages[message_index] = java_message
            g_failed_job_java_message_types[message_index] = java_message_type
        else:
            g_success_jobs.append(testname)
            g_success_job_java_messages.append(java_message)
            g_success_job_java_message_types.append(java_message_type)

def extract_java_messages():
    if False:
        i = 10
        return i + 15
    '\n    loop through java_*_0.out.txt and extract potentially dangerous WARN/ERRR/FATAL\n    messages associated with a test.  The test may even pass but something terrible\n    has actually happened.\n\n    :return: none\n    '
    global g_jenkins_url
    global g_failed_test_info_dict
    global g_java_filenames
    global g_failed_jobs
    global g_failed_job_java_messages
    global g_failed_job_java_message_types
    global g_success_jobs
    global g_success_job_java_messages
    global g_success_job_java_message_types
    global g_java_general_bad_messages
    global g_java_general_bad_message_types
    if len(g_failed_jobs) > 0:
        for fname in g_java_filenames:
            temp_strings = fname.split('/')
            start_url = g_jenkins_url
            for windex in range(6, len(temp_strings)):
                start_url = os.path.join(start_url, temp_strings[windex])
            try:
                get_console_out(start_url)
                grab_java_message()
            except:
                pass
    if len(g_failed_jobs) > 0:
        g_failed_test_info_dict['failed_tests_info *********'] = [g_failed_jobs, g_failed_job_java_messages, g_failed_job_java_message_types]
    if len(g_success_jobs) > 0:
        g_failed_test_info_dict['passed_tests_info *********'] = [g_success_jobs, g_success_job_java_messages, g_success_job_java_message_types]
    if len(g_java_general_bad_messages) > 0:
        g_failed_test_info_dict['9.general_bad_java_messages'] = [g_java_general_bad_messages, g_java_general_bad_message_types]

def save_dict():
    if False:
        i = 10
        return i + 15
    '\n    Save the log scraping results into logs denoted by g_output_filename_failed_tests and\n    g_output_filename_passed_tests.\n\n    :return: none\n    '
    global g_test_root_dir
    global g_output_filename_failed_tests
    global g_output_filename_passed_tests
    global g_output_pickle_filename
    global g_failed_test_info_dict
    if '2.build_id' not in g_failed_test_info_dict.keys():
        g_failed_test_info_dict['2.build_id'] = 'unknown'
    build_id = g_failed_test_info_dict['2.build_id']
    g_output_filename_failed_tests = g_output_filename_failed_tests + '_build_' + build_id + '_failed_tests.log'
    g_output_filename_passed_tests = g_output_filename_passed_tests + '_build_' + build_id + '_passed_tests.log'
    g_output_pickle_filename = g_output_pickle_filename + '_build_' + build_id + '.pickle'
    allKeys = sorted(g_failed_test_info_dict.keys())
    with open(g_output_pickle_filename, 'wb') as test_file:
        pickle.dump(g_failed_test_info_dict, test_file)
    text_file_failed_tests = open(g_output_filename_failed_tests, 'w')
    text_file_passed_tests = None
    allKeys = sorted(g_failed_test_info_dict.keys())
    write_passed_tests = False
    if 'passed_tests_info *********' in allKeys:
        text_file_passed_tests = open(g_output_filename_passed_tests, 'w')
        write_passed_tests = True
    for keyName in allKeys:
        val = g_failed_test_info_dict[keyName]
        if isinstance(val, list):
            if len(val) == 3:
                if keyName == 'failed_tests_info *********':
                    write_test_java_message(keyName, val, text_file_failed_tests)
                if keyName == 'passed_tests_info *********':
                    write_test_java_message(keyName, val, text_file_passed_tests)
            elif len(val) == 2:
                write_java_message(keyName, val, text_file_failed_tests)
                if write_passed_tests:
                    write_java_message(keyName, val, text_file_passed_tests)
        else:
            write_general_build_message(keyName, val, text_file_failed_tests)
            if write_passed_tests:
                write_general_build_message(keyName, val, text_file_passed_tests)
    text_file_failed_tests.close()
    if write_passed_tests:
        text_file_passed_tests.close()

def write_general_build_message(key, val, text_file):
    if False:
        while True:
            i = 10
    '\n    Write key/value into log file when the value is a string and not a list.\n\n    Parameters\n    ----------\n    key :  str\n        key value in g_failed_test_info_dict\n    value :  str\n        corresponding value associated with the key in key\n    text_file : file handle\n        file handle of log file to write the info to.\n\n\n    :return: none\n    '
    text_file.write(key + ': ')
    text_file.write(val)
    text_file.write('\n\n')

def write_test_java_message(key, val, text_file):
    if False:
        while True:
            i = 10
    '\n   Write key/value into log file when the value is a list of strings\n   or even a list of list of string.  These lists are associated with\n   unit tests that are executed in the jenkins job.\n\n    Parameters\n    ----------\n    key :  str\n        key value in g_failed_test_info_dict\n    value :  list of str or list of list of str\n        corresponding value associated with the key in key\n    text_file : file handle\n        file handle of log file to write the info to.\n\n   :return: none\n   '
    global g_failed_jobs
    text_file.write(key)
    text_file.write('\n')
    for index in range(len(val[0])):
        if val[0][index] in g_failed_jobs or (val[0][index] not in g_failed_jobs and len(val[1][index]) > 0):
            text_file.write('\nTest Name: ')
            text_file.write(val[0][index])
            text_file.write('\n')
        if len(val[1][index]) > 0 and len(val) >= 3:
            text_file.write('Java Message Type and Message: \n')
            for eleIndex in range(len(val[1][index])):
                text_file.write(val[2][index][eleIndex] + ' ')
                text_file.write(val[1][index][eleIndex])
                text_file.write('\n\n')
    text_file.write('\n')
    text_file.write('\n')

def update_summary_file():
    if False:
        for i in range(10):
            print('nop')
    '\n    Concatecate all log file into a summary text file to be sent to users\n    at the end of a daily log scraping.\n\n    :return: none\n    '
    global g_summary_text_filename
    global g_output_filename_failed_tests
    global g_output_filename_passed_tests
    with open(g_summary_text_filename, 'a') as tempfile:
        write_file_content(tempfile, g_output_filename_failed_tests)
        write_file_content(tempfile, g_output_filename_passed_tests)

def write_file_content(fhandle, file2read):
    if False:
        print('Hello World!')
    '\n    Write one log file into the summary text file.\n\n    Parameters\n    ----------\n    fhandle :  Python file handle\n        file handle to the summary text file\n    file2read : Python file handle\n        file handle to log file where we want to add its content to the summary text file.\n\n    :return: none\n    '
    if os.path.isfile(file2read):
        with open(file2read, 'r') as tfile:
            fhandle.write('============ Content of ' + file2read)
            fhandle.write('\n')
            fhandle.write(tfile.read())
            fhandle.write('\n\n')

def write_java_message(key, val, text_file):
    if False:
        for i in range(10):
            print('nop')
    '\n    Loop through all java messages that are not associated with a unit test and\n    write them into a log file.\n\n    Parameters\n    ----------\n    key :  str\n        9.general_bad_java_messages\n    val : list of list of str\n        contains the bad java messages and the message types.\n\n\n    :return: none\n    '
    text_file.write(key)
    text_file.write('\n')
    if len(val[0]) > 0 and len(val) >= 3:
        for index in range(len(val[0])):
            text_file.write('Java Message Type: ')
            text_file.write(val[1][index])
            text_file.write('\n')
            text_file.write('Java Message: ')
            for jmess in val[2][index]:
                text_file.write(jmess)
                text_file.write('\n')
        text_file.write('\n \n')

def load_java_messages_to_ignore():
    if False:
        while True:
            i = 10
    '\n    Load in pickle file that contains dict structure with bad java messages to ignore per unit test\n    or for all cases.  The ignored bad java info is stored in g_ok_java_messages dict.\n\n    :return:\n    '
    global g_ok_java_messages
    global g_java_message_pickle_filename
    if os.path.isfile(g_java_message_pickle_filename):
        with open(g_java_message_pickle_filename, 'rb') as tfile:
            g_ok_java_messages = pickle.load(tfile)
    else:
        g_ok_java_messages['general'] = []

def main(argv):
    if False:
        print('Hello World!')
    '\n    Main program.\n\n    @return: none\n    '
    global g_script_name
    global g_test_root_dir
    global g_temp_filename
    global g_output_filename_failed_tests
    global g_output_filename_passed_tests
    global g_output_pickle_filename
    global g_failure_occurred
    global g_failed_test_info_dict
    global g_java_message_pickle_filename
    global g_summary_text_filename
    if len(argv) < 3:
        print('Must resource url like http://mr-0xa1:8080/view/wendy_jenkins/job/h2o_regression_pyunit_medium_large/lastBuild/consoleFull, filename of summary text, filename (optional ending in .pickle) to retrieve Java error messages to exclude.\n')
        sys.exit(1)
    else:
        g_script_name = os.path.basename(argv[0])
        resource_url = argv[1]
        g_temp_filename = os.path.join(g_test_root_dir, 'tempText')
        g_summary_text_filename = os.path.join(g_test_root_dir, argv[2])
        if len(argv) == 4:
            g_java_message_pickle_filename = argv[3]
        get_console_out(resource_url)
        extract_job_build_url(resource_url)
        log_filename = g_failed_test_info_dict['1.jobName']
        log_pickle_filename = g_failed_test_info_dict['1.jobName']
        g_java_message_pickle_filename = os.path.join(g_test_root_dir, g_java_message_pickle_filename)
        g_output_filename_failed_tests = os.path.join(g_test_root_dir, log_filename)
        g_output_filename_passed_tests = os.path.join(g_test_root_dir, log_filename)
        g_output_pickle_filename = os.path.join(g_test_root_dir, log_pickle_filename)
        load_java_messages_to_ignore()
        extract_test_results()
        extract_java_messages()
        if len(g_failed_jobs) > 0 or g_failed_test_info_dict['7.build_failure'] == 'Yes':
            g_failure_occurred = True
        if g_failure_occurred:
            save_dict()
            update_summary_file()
            print(g_failed_test_info_dict['1.jobName'] + ' build ' + g_failed_test_info_dict['2.build_id'] + ',')
        else:
            print('')
if __name__ == '__main__':
    main(sys.argv)