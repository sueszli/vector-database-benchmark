import os
import sys
import json
import functools
import subprocess
from urllib.request import urlopen
from urllib.request import Request
from urllib.error import HTTPError, URLError
from sparktestsupport import SPARK_HOME, ERROR_CODES
from sparktestsupport.shellutils import run_cmd

def print_err(msg):
    if False:
        return 10
    '\n    Given a set of arguments, will print them to the STDERR stream\n    '
    print(msg, file=sys.stderr)

def post_message_to_github(msg, ghprb_pull_id):
    if False:
        while True:
            i = 10
    print('Attempting to post to GitHub...')
    api_url = os.getenv('GITHUB_API_BASE', 'https://api.github.com/repos/apache/spark')
    url = api_url + '/issues/' + ghprb_pull_id + '/comments'
    github_oauth_key = os.environ['GITHUB_OAUTH_KEY']
    posted_message = json.dumps({'body': msg})
    request = Request(url, headers={'Authorization': 'token %s' % github_oauth_key, 'Content-Type': 'application/json'}, data=posted_message.encode('utf-8'))
    try:
        response = urlopen(request)
        if response.getcode() == 201:
            print(' > Post successful.')
    except HTTPError as http_e:
        print_err('Failed to post message to GitHub.')
        print_err(' > http_code: %s' % http_e.code)
        print_err(' > api_response: %s' % http_e.read())
        print_err(' > data: %s' % posted_message)
    except URLError as url_e:
        print_err('Failed to post message to GitHub.')
        print_err(' > urllib_status: %s' % url_e.reason[1])
        print_err(' > data: %s' % posted_message)

def pr_message(build_display_name, build_url, ghprb_pull_id, short_commit_hash, commit_url, msg, post_msg=''):
    if False:
        for i in range(10):
            print('nop')
    str_args = (build_display_name, msg, build_url, ghprb_pull_id, short_commit_hash, commit_url, str(' ' + post_msg + '.') if post_msg else '.')
    return '**[Test build %s %s](%stestReport)** for PR %s at commit [`%s`](%s)%s' % str_args

def run_pr_checks(pr_tests, ghprb_actual_commit, sha1):
    if False:
        for i in range(10):
            print('nop')
    '\n    Executes a set of pull request checks to ease development and report issues with various\n    components such as style, linting, dependencies, compatibilities, etc.\n    @return a list of messages to post back to GitHub\n    '
    current_pr_head = run_cmd(['git', 'rev-parse', 'HEAD'], return_output=True).strip()
    pr_results = list()
    for pr_test in pr_tests:
        test_name = pr_test + '.sh'
        pr_results.append(run_cmd(['bash', os.path.join(SPARK_HOME, 'dev', 'tests', test_name), ghprb_actual_commit, sha1], return_output=True).rstrip())
        run_cmd(['git', 'checkout', '-f', current_pr_head])
    return pr_results

def run_tests(tests_timeout):
    if False:
        print('Hello World!')
    '\n    Runs the `dev/run-tests` script and responds with the correct error message\n    under the various failure scenarios.\n    @return a tuple containing the test result code and the result note to post to GitHub\n    '
    test_result_code = subprocess.Popen(['timeout', tests_timeout, os.path.join(SPARK_HOME, 'dev', 'run-tests')]).wait()
    failure_note_by_errcode = {1: 'executing the `dev/run-tests` script', ERROR_CODES['BLOCK_GENERAL']: 'some tests', ERROR_CODES['BLOCK_RAT']: 'RAT tests', ERROR_CODES['BLOCK_SCALA_STYLE']: 'Scala style tests', ERROR_CODES['BLOCK_JAVA_STYLE']: 'Java style tests', ERROR_CODES['BLOCK_PYTHON_STYLE']: 'Python style tests', ERROR_CODES['BLOCK_R_STYLE']: 'R style tests', ERROR_CODES['BLOCK_DOCUMENTATION']: 'to generate documentation', ERROR_CODES['BLOCK_BUILD']: 'to build', ERROR_CODES['BLOCK_BUILD_TESTS']: 'build dependency tests', ERROR_CODES['BLOCK_MIMA']: 'MiMa tests', ERROR_CODES['BLOCK_SPARK_UNIT_TESTS']: 'Spark unit tests', ERROR_CODES['BLOCK_PYSPARK_UNIT_TESTS']: 'PySpark unit tests', ERROR_CODES['BLOCK_PYSPARK_PIP_TESTS']: 'PySpark pip packaging tests', ERROR_CODES['BLOCK_SPARKR_UNIT_TESTS']: 'SparkR unit tests', ERROR_CODES['BLOCK_TIMEOUT']: 'from timeout after a configured wait of `%s`' % tests_timeout}
    if test_result_code == 0:
        test_result_note = ' * This patch passes all tests.'
    else:
        note = failure_note_by_errcode.get(test_result_code, 'due to an unknown error code, %s' % test_result_code)
        test_result_note = ' * This patch **fails %s**.' % note
    return [test_result_code, test_result_note]

def main():
    if False:
        for i in range(10):
            print('nop')
    ghprb_pull_id = os.environ['ghprbPullId']
    ghprb_actual_commit = os.environ['ghprbActualCommit']
    ghprb_pull_title = os.environ['ghprbPullTitle'].lower()
    sha1 = os.environ['sha1']
    os.environ['SPARK_JENKINS_PRB'] = 'true'
    if 'test-maven' in ghprb_pull_title:
        os.environ['SPARK_JENKINS_BUILD_TOOL'] = 'maven'
    if 'test-hadoop3' in ghprb_pull_title:
        os.environ['SPARK_JENKINS_BUILD_PROFILE'] = 'hadoop3'
    if 'test-scala2.13' in ghprb_pull_title:
        os.environ['SPARK_JENKINS_BUILD_SCALA_PROFILE'] = 'scala2.13'
    build_display_name = os.environ['BUILD_DISPLAY_NAME']
    build_url = os.environ['BUILD_URL']
    project_url = os.getenv('SPARK_PROJECT_URL', 'https://github.com/apache/spark')
    commit_url = project_url + '/commit/' + ghprb_actual_commit
    short_commit_hash = ghprb_actual_commit[0:7]
    tests_timeout = '500m'
    pr_tests = ['pr_merge_ability', 'pr_public_classes']
    github_message = functools.partial(pr_message, build_display_name, build_url, ghprb_pull_id, short_commit_hash, commit_url)
    post_message_to_github(github_message('has started'), ghprb_pull_id)
    pr_check_results = run_pr_checks(pr_tests, ghprb_actual_commit, sha1)
    (test_result_code, test_result_note) = run_tests(tests_timeout)
    result_message = github_message('has finished')
    result_message += '\n' + test_result_note + '\n'
    result_message += '\n'.join(pr_check_results)
    post_message_to_github(result_message, ghprb_pull_id)
    sys.exit(test_result_code)
if __name__ == '__main__':
    main()