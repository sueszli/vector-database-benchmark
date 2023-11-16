"""apport package hook for Bazaar"""
from apport.hookutils import *
import os
bzr_log = os.path.expanduser('~/.bzr.log')
dot_bzr = os.path.expanduser('~/.bazaar')

def _add_log_tail(report):
    if False:
        i = 10
        return i + 15
    if 'BzrLogTail' in report:
        return
    bzr_log_lines = open(bzr_log).readlines()
    bzr_log_lines.reverse()
    bzr_log_tail = []
    blanks = 0
    for line in bzr_log_lines:
        if line == '\n':
            blanks += 1
        bzr_log_tail.append(line)
        if blanks >= 2:
            break
    bzr_log_tail.reverse()
    report['BzrLogTail'] = ''.join(bzr_log_tail)

def add_info(report):
    if False:
        for i in range(10):
            print('nop')
    _add_log_tail(report)
    if 'BzrPlugins' not in report:
        report['BzrPlugins'] = command_output(['bzr', 'plugins', '-v'])
    report['CrashDB'] = 'bzr'