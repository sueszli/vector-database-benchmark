"""This script checks if backend overall line coverage is 100%."""
from __future__ import annotations
import os
import re
import subprocess
import sys
from scripts import common

def main() -> None:
    if False:
        return 10
    'Checks if backend overall line coverage is 100%.'
    env = os.environ.copy()
    cmd = [sys.executable, '-m', 'coverage', 'report', '--omit="%s*","third_party/*","/usr/share/*"' % common.OPPIA_TOOLS_DIR, '--show-missing']
    process = subprocess.run(cmd, capture_output=True, encoding='utf-8', env=env, check=False)
    if process.stdout.strip() == 'No data to report.':
        raise RuntimeError('Run backend tests before running this script. ' + '\nOUTPUT: %s\nERROR: %s' % (process.stdout, process.stderr))
    if process.returncode:
        raise RuntimeError('Failed to calculate coverage because subprocess failed. ' + '\nOUTPUT: %s\nERROR: %s' % (process.stdout, process.stderr))
    print(process.stdout)
    coverage_result = re.search('TOTAL\\s+(\\d+)\\s+(?P<total>\\d+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)%\\s+', process.stdout)
    uncovered_lines = -1.0
    if coverage_result:
        uncovered_lines = float(coverage_result.group('total'))
    else:
        raise RuntimeError('Error in parsing coverage report.')
    if uncovered_lines != 0:
        print('--------------------------------------------')
        print('Backend overall line coverage checks failed.')
        print('--------------------------------------------')
        sys.exit(1)
    else:
        print('--------------------------------------------')
        print('Backend overall line coverage checks passed.')
        print('--------------------------------------------')
if __name__ == '__main__':
    main()