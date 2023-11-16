import re
import sys
from subprocess import run
from typing import NoReturn

def ProtectBranches() -> NoReturn:
    if False:
        return 10
    hookid = 'protect-branches'
    protected_branches = ['main', 'branch-\\d+\\.\\d+']
    current_branch = run(['git', 'branch', '--show-current'], capture_output=True).stdout.decode(sys.stdout.encoding).replace('\n', '')
    for branch in protected_branches:
        regex = re.compile(branch)
        if regex.match(current_branch):
            print(f"\nYou were about to push to `{current_branch}`, which is disallowed by default.\nIf that's really what you intend, run the following command:\n\n        SKIP={hookid} git push\n")
            sys.exit(1)
    sys.exit(0)
if __name__ == '__main__':
    ProtectBranches()