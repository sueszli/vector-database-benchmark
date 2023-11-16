import sys
import os
import json
import pickle
import copy
import subprocess
'\nThis script is written to grab the logs from our GLRM runs and saved them onto our local machines for\nlater analysis.  For me, this is how I will call this script and the input arguments I will use:\n\npython grabGLRMrunLogs.py /Users/wendycwong/Documents/PUBDEV_3454_GLRM/experimentdata/glrm_memory_10_25_16\n    http://mr-0xa1:8080/view/wendy/job/glrm_memory_performance/\n    /Users/wendycwong/Documents/PUBDEV_3454_GLRM/experimentdata/glrm_memory_10_25_16\n    java_0_0.out_airline.txt java_1_0.out_milsongs.txt pyunit_airlines_performance_profile.py.out.txt\n    pyunit_milsongs_performance_profile.py.out.txt 8 26\n'
g_test_root_dir = os.path.dirname(os.path.realpath(__file__))
g_airline_py_tail = '/artifact/h2o-py/GLRM_performance_tests/results/pyunit_airlines_performance_profile.py.out.txt'
g_milsongs_py_tail = '/artifact/h2o-py/GLRM_performance_tests/results/pyunit_milsong_performance_profile.py.out.txt'
g_airline_java_tail = '/artifact/h2o-py/GLRM_performance_tests/results/java_0_0.out.txt'
g_milsongs_java_tail = '/artifact/h2o-py/GLRM_performance_tests/results/java_1_0.out.txt'
g_log_base_dir = ''
g_airline_java = ''
g_milsongs_java = ''
g_airline_python = ''
g_milsongs_python = ''
g_jenkins_url = ''
g_start_build_number = 0
g_end_build_number = 1

def get_file_out(build_index, python_name, jenkin_name):
    if False:
        while True:
            i = 10
    '\n    This function will grab one log file from Jenkins and save it to local user directory\n    :param g_jenkins_url:\n    :param build_index:\n    :param airline_java:\n    :param airline_java_tail:\n    :return:\n    '
    global g_log_base_dir
    global g_jenkins_url
    global g_log_base_dir
    directoryB = g_log_base_dir + '/Build' + str(build_index)
    if not os.path.isdir(directoryB):
        os.mkdir(directoryB)
    url_string_full = g_jenkins_url + '/' + str(build_index) + jenkin_name
    filename = os.path.join(directoryB, python_name)
    full_command = 'curl ' + url_string_full + ' > ' + filename
    subprocess.call(full_command, shell=True)

def main(argv):
    if False:
        while True:
            i = 10
    '\n    Main program.\n\n    @return: none\n    '
    global g_log_base_dir
    global g_airline_java
    global g_milsongs_java
    global g_airline_python
    global g_milsongs_python
    global g_jenkins_url
    global g_airline_py_tail
    global g_milsongs_py_tail
    global g_airline_java_tail
    global g_milsongs_java_tail
    if len(argv) < 9:
        print('python grabGLRMrunLogs logsBaseDirectory airlineJavaFileNameWithPath milsongJavaFileNameWithPath ' + 'airlinePyunitWithPath airlinePyunitWithPath jenkinsJobURL startBuild# endBuild#.\n')
        sys.exit(1)
    else:
        g_log_base_dir = argv[1]
        g_jenkins_url = argv[2]
        g_airline_java = argv[3]
        g_milsongs_java = argv[4]
        g_airline_python = argv[5]
        g_milsongs_python = argv[6]
        start_number = int(argv[7])
        end_number = int(argv[8])
        if start_number > end_number:
            print('startBuild# must be <= end_number')
            sys.exit(1)
        else:
            for build_index in range(start_number, end_number + 1):
                get_file_out(build_index, g_airline_java, g_airline_java_tail)
                get_file_out(build_index, g_milsongs_java, g_milsongs_java_tail)
                get_file_out(build_index, g_airline_python, g_airline_py_tail)
                get_file_out(build_index, g_milsongs_python, g_milsongs_py_tail)
if __name__ == '__main__':
    main(sys.argv)