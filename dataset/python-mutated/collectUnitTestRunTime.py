import sys
import os
import json
import subprocess
import time
import numpy as np
"\nThis script is written to run Runit or Pyunit tests repeatedly for N times in order to\ncollect run time statistics especially for tests running with randomly generated datasets\nand parameter values.  You should invoke this script using the following commands:\n\npython collectUnitTestRunTime.py 10 'python run.py_path/run.py --wipe --test dir_to_test/test1,\npython run.py_path/run.py --wipe --test dir_to_test2/test2,...' True\n\nThe above command will run all the commands in the list 10 times and collect the run time printed out\nby the run.  Make sure there is no space between each command.  They are only\n separated by comma.\nIt will collect all the run time as a sequence, calculate the min/max/mean/std and store\nthe info as a json file with the testname in the directory where you are running this script.  Since the\nlast argument is True, it will add the current run result to the old ones run and stored already.  The\nstatistics calcluated will be over all the runs, old ones and new ones.  This is done to help you break up\nyour experiments into chunks instead of having to run it at one time.  If you are doing a fresh run, put False\nas the last argument\n"
g_test_root_dir = os.path.dirname(os.path.realpath(__file__))
g_temp_filename = os.path.join(g_test_root_dir, 'tempText')
g_test_result_start = 'PASS'

def run_commands(command, number_to_run, temp_file):
    if False:
        i = 10
        return i + 15
    '\n    This function will run the command for number_to_run number of times.  For each run,\n    it store the screen print out into temp_file.  After finishing a run, it will go to\n    temp_file and collect the run time information and store it in a list in result_dict\n    which is a python dict structure.  The result_dict will be returned.\n\n    :param command: string containing the command to be run\n    :param number_to_run: integer denoting the number of times to run command\n    :param temp_file: string containing file name to store run command output\n\n    :return: result_dict: python dict containing run time in secs as a list\n    '
    result_dict = dict()
    temp_string = command.split()
    testname = temp_string[-1]
    temp_string = testname.split('/')
    result_dict['test_name'] = temp_string[-1]
    result_dict['run_time_secs'] = []
    full_command = command + ' > ' + temp_file
    for run_index in range(0, number_to_run):
        child = subprocess.Popen(full_command, shell=True)
        while child.poll() is None:
            time.sleep(10)
        with open(temp_file, 'r') as thefile:
            for each_line in thefile:
                print(each_line)
                temp_string = each_line.split()
                if len(temp_string) > 0:
                    if temp_string[0] == 'PASS':
                        test_time = temp_string[2]
                        try:
                            runtime = test_time[:-1]
                            result_dict['run_time_secs'].append(float(runtime))
                        except:
                            print('Cannot convert run time.  It is {0}\n'.format(runtime))
                        break
    return result_dict

def write_result_summary(result_dict, directory_path, is_new_run):
    if False:
        for i in range(10):
            print('nop')
    '\n    This function will summarize the run time in secs and store the run result in a json file.\n    In addition, if is_new_run = False, results will be read from an old json file.  The final\n    result summary will be a combination of the old run and new run.\n\n    :param result_dict: python dict, contains all run time results\n    :param directory_path: string containing the directory path where we are going to\n    store the run result json file\n    :param is_new_run: bool, denoting whether this is a fresh run if True and vice versa\n\n    :return: None\n    '
    dict_keys = list(result_dict)
    if 'test_name' in dict_keys:
        json_file = os.path.join(directory_path, result_dict['test_name'] + '.json')
        run_time = []
        if os.path.exists(json_file) and (not is_new_run):
            with open(json_file, 'r') as test_file:
                temp_dict = json.load(test_file)
                run_time = temp_dict['run_time_secs']
        if 'run_time_secs' in dict_keys:
            if len(run_time) > 0:
                run_time.extend(result_dict['run_time_secs'])
            else:
                run_time = result_dict['run_time_secs']
            if len(run_time) > 0:
                result_dict['max_run_time_secs'] = max(run_time)
                result_dict['min_run_time_secs'] = min(run_time)
                result_dict['mean_run_time_secs'] = np.mean(run_time)
                result_dict['run_time_std'] = np.std(run_time)
                result_dict['total_number_of_runs'] = len(run_time)
                result_dict['run_time_secs'] = run_time
                with open(json_file, 'w') as test_file:
                    json.dump(result_dict, test_file)
                print('Run result summary: \n {0}'.format(result_dict))
        else:
            print('Your result summary dictionary does not contain run time data!\n')
    else:
        print('Cannot find your test name.  Nothing is done.\n')

def main(argv):
    if False:
        for i in range(10):
            print('nop')
    '\n    Main program.  Take user input, parse it and call other functions to execute the commands\n    and extract run summary and store run result in json file\n\n    @return: none\n    '
    global g_test_root_dir
    global g_temp_filename
    if len(argv) < 3:
        print("invoke this script as python collectUnitTestRunTime.py 10 'python run.py_path/run.py --wipe --test dir_to_test/test1,python run.py_path/run.py --wipe --test dir_to_test2/test2,...' True\n")
        sys.exit(1)
    else:
        repeat_number = int(argv[1])
        command_lists = argv[2]
        if argv[3] == 'True':
            is_new_run = True
        else:
            is_new_run = False
        for command in command_lists.split(','):
            result_dict = run_commands(command, repeat_number, g_temp_filename)
            write_result_summary(result_dict, g_test_root_dir, is_new_run)
if __name__ == '__main__':
    main(sys.argv)