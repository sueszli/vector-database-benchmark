#!/usr/bin/env python3

# SPDX-FileCopyrightText: Florian Bruhin (The Compiler) <mail@qutebrowser.org>
#
# SPDX-License-Identifier: GPL-3.0-or-later


"""Check if docs changed and output an error if so."""

import sys
import subprocess
import os
import os.path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir,
                                os.pardir))

from scripts import utils

code = subprocess.run(['git', '--no-pager', 'diff', '--exit-code', '--stat',
                       '--', 'doc'], check=False).returncode

if os.environ.get('GITHUB_REF', 'refs/heads/main') != 'refs/heads/main':
    if code != 0:
        print("Docs changed but ignoring change as we're building a PR")
    sys.exit(0)

if code != 0:
    print()
    print('The autogenerated docs changed, please run this to update them:')
    print('   tox -e docs')
    print('   git commit -am "Update docs"')
    print()
    print('(Or you have uncommitted changes, in which case you can ignore '
          'this.)')
    if utils.ON_CI:
        utils.gha_error('The autogenerated docs changed')
        print()
        with utils.gha_group('Diff'):
            subprocess.run(['git', '--no-pager', 'diff'], check=True)
sys.exit(code)
