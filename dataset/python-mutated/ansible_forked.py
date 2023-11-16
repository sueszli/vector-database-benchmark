"""Run each test in its own fork. PYTEST_DONT_REWRITE"""
from __future__ import annotations
import os
import pickle
import tempfile
import warnings
from pytest import Item, hookimpl, TestReport
from _pytest.runner import runtestprotocol

@hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    if False:
        while True:
            i = 10
    'Entry point for enabling this plugin.'
    warnings.filterwarnings('ignore', '^This process .* is multi-threaded, use of .* may lead to deadlocks in the child.$', DeprecationWarning)
    item_hook = item.ihook
    item_hook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    reports = run_item(item, nextitem)
    for report in reports:
        item_hook.pytest_runtest_logreport(report=report)
    item_hook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    return True

def run_item(item, nextitem):
    if False:
        return 10
    'Run the item in a child process and return a list of reports.'
    with tempfile.NamedTemporaryFile() as temp_file:
        pid = os.fork()
        if not pid:
            temp_file.delete = False
            run_child(item, nextitem, temp_file.name)
        return run_parent(item, pid, temp_file.name)

def run_child(item, nextitem, result_path):
    if False:
        while True:
            i = 10
    'Run the item, record the result and exit. Called in the child process.'
    with warnings.catch_warnings(record=True) as captured_warnings:
        reports = runtestprotocol(item, nextitem=nextitem, log=False)
    with open(result_path, 'wb') as result_file:
        pickle.dump((reports, captured_warnings), result_file)
    os._exit(0)

def run_parent(item, pid, result_path):
    if False:
        i = 10
        return i + 15
    'Wait for the child process to exit and return the test reports. Called in the parent process.'
    exit_code = waitstatus_to_exitcode(os.waitpid(pid, 0)[1])
    if exit_code:
        reason = 'Test CRASHED with exit code {}.'.format(exit_code)
        report = TestReport(item.nodeid, item.location, {x: 1 for x in item.keywords}, 'failed', reason, 'call', user_properties=item.user_properties)
        if item.get_closest_marker('xfail'):
            report.outcome = 'skipped'
            report.wasxfail = reason
        reports = [report]
    else:
        with open(result_path, 'rb') as result_file:
            (reports, captured_warnings) = pickle.load(result_file)
        for warning in captured_warnings:
            warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)
    return reports

def waitstatus_to_exitcode(status):
    if False:
        print('Hello World!')
    'Convert a wait status to an exit code.'
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return -os.WTERMSIG(status)
    raise ValueError(status)