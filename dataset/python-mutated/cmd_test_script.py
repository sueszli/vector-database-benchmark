"""Front-end command for shell-like test scripts.

See doc/developers/testing.txt for more explanations.
This module should be importable even if testtools aren't available.
"""
from __future__ import absolute_import
import os
from bzrlib import commands, option

class cmd_test_script(commands.Command):
    """Run a shell-like test from a file."""
    hidden = True
    takes_args = ['infile']
    takes_options = [option.Option('null-output', help='Null command outputs match any output.')]

    @commands.display_command
    def run(self, infile, null_output=False):
        if False:
            for i in range(10):
                print('nop')
        from bzrlib import tests
        from bzrlib.tests.script import TestCaseWithTransportAndScript
        f = open(infile)
        try:
            script = f.read()
        finally:
            f.close()

        class Test(TestCaseWithTransportAndScript):
            script = None

            def test_it(self):
                if False:
                    while True:
                        i = 10
                self.run_script(script, null_output_matches_anything=null_output)
        runner = tests.TextTestRunner(stream=self.outf)
        test = Test('test_it')
        test.path = os.path.realpath(infile)
        res = runner.run(test)
        return len(res.errors) + len(res.failures)