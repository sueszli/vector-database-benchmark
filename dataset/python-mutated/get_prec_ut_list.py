"""To get a list of prec ut """
import os

def get_prec_ut_list(all_test_cases, prec_test_cases):
    if False:
        i = 10
        return i + 15
    'Select the ut that needs to be executed'
    all_test_cases_list = all_test_cases.strip().split('\n')
    prec_test_cases_list = prec_test_cases.strip().split('\n')
    all_test_cases_list_new = [item.rstrip() for item in all_test_cases_list]
    prec_test_cases_list_new = [item.rstrip() for item in prec_test_cases_list]
    if len(prec_test_cases) == 0:
        return
    case_to_run = ['test_prec_ut']
    for case in all_test_cases_list_new:
        if case in prec_test_cases_list_new:
            case_to_run.append(case)
        else:
            print(f'{case} will not run in PRECISION_TEST mode.')
    with open(file_path, 'w') as f:
        f.write('\n'.join(case_to_run))
if __name__ == '__main__':
    with open('ut_list', 'r') as f:
        prec_test_cases = f.read()
    BUILD_DIR = os.getcwd()
    file_path = os.path.join(BUILD_DIR, 'all_ut_list')
    with open(file_path, 'r') as f:
        all_test_cases = f.read()
    get_prec_ut_list(all_test_cases, prec_test_cases)