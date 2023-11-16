import sys
import os
import json
import subprocess
import time
import numpy as np
"\nThis script is written to run Runit or Pyunit tests repeatedly for N times in order to\ncollect run time especially for tests running with randomly generated datasets\nand parameter values.  You should invoke this script using the following commands:\n\npython findLongUnitTestRunTime.py 10 'python run.py_path/run.py --wipe --test dir_to_test/test1,\npython run.py_path/run.py --wipe --test dir_to_test2/test2,...' True\n\nThe above command will run all the commands in the list 10 times and collect the run time printed out\nby the run.  Make sure there is no space between each command.  They are only\n separated by comma.\n\nIt will collect the unit test run time and will stop the operation if the test takes too long to run.\nIt will basically save all the data used to train the unit tests and only wipe them if they run shorter\nthan a time limit that we set.  It will move onto the next test in the list if it does find a long running\ntest.\n"
g_test_root_dir = os.path.dirname(os.path.realpath(__file__))
g_temp_filename = os.path.join(g_test_root_dir, 'tempText')
g_test_result_start = 'PASS'
g_max_runtime_secs = 4000
g_finished_this_unit_test = False

def run_commands(command, number_to_run, temp_file):
    if False:
        return 10
    '\n    This function will run the command for number_to_run number of times.  For each run,\n    it will capture the run time for the unit test and will move on to the next test if\n    it takes too long to run this one.\n\n    :param command: string containing the command to be run\n    :param number_to_run: integer denoting the number of times to run command\n    :param temp_file: string containing file name to store run command output\n\n    :return: None\n    '
    global g_max_runtime_secs
    global g_finished_this_unit_test
    temp_string = command.split()
    testname = temp_string[-1]
    temp_string = testname.split('/')
    full_command = command + ' > ' + temp_file
    g_finished_this_unit_test = False
    for run_index in range(0, number_to_run):
        if g_finished_this_unit_test:
            break
        child = subprocess.Popen(full_command, shell=True)
        while child.poll() is None:
            time.sleep(20)
        with open(temp_file, 'r') as thefile:
            for each_line in thefile:
                temp_string = each_line.split()
                if len(temp_string) > 0:
                    if temp_string[0] == 'PASS':
                        test_time = temp_string[2]
                        try:
                            runtime = float(test_time[:-1])
                            print('Unit test run time is {0}'.format(runtime))
                            if runtime > g_max_runtime_secs:
                                g_finished_this_unit_test = True
                        except:
                            print('Cannot convert run time.  It is {0}\n'.format(runtime))
                        break

def main(argv):
    if False:
        print('Hello World!')
    '\n    Main program.  Take user input, parse it and call other functions to execute the commands\n    and find long running unit tests, store the dataset and parameter settings and move onto\n    the next unit tests if applicable.\n\n    @return: none\n    '
    global g_test_root_dir
    global g_temp_filename
    if len(argv) < 2:
        print("invoke this script as python collectUnitTestRunTime.py 10 'python run.py_path/run.py --wipe --test dir_to_test/test1,python run.py_path/run.py --wipe --test dir_to_test2/test2,...' True\n")
        sys.exit(1)
    else:
        repeat_number = int(argv[1])
        command_lists = argv[2]
        for command in command_lists.split(','):
            run_commands(command, repeat_number, g_temp_filename)
if __name__ == '__main__':
    main(sys.argv)