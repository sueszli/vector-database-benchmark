from __future__ import absolute_import
from __future__ import print_function
from buildbot_worker.test.util import command

class SourceCommandTestMixin(command.CommandTestMixin):
    """
    Support for testing Source Commands; an extension of CommandTestMixin
    """

    def make_command(self, cmdclass, args, makedirs=False, initial_sourcedata=''):
        if False:
            print('Hello World!')
        "\n        Same as the parent class method, but this also adds some source-specific\n        patches:\n\n        * writeSourcedata - writes to self.sourcedata (self is the TestCase)\n        * readSourcedata - reads from self.sourcedata\n        * doClobber - invokes RunProcess(0, ['clobber', DIRECTORY])\n        * doCopy - invokes RunProcess(0, ['copy', cmd.srcdir, cmd.workdir])\n        "
        cmd = command.CommandTestMixin.make_command(self, cmdclass, args, makedirs)
        self.sourcedata = initial_sourcedata

        def readSourcedata():
            if False:
                while True:
                    i = 10
            if self.sourcedata is None:
                raise IOError('File not found')
            return self.sourcedata
        cmd.readSourcedata = readSourcedata

        def writeSourcedata(res):
            if False:
                for i in range(10):
                    print('nop')
            self.sourcedata = cmd.sourcedata
            return res
        cmd.writeSourcedata = writeSourcedata

    def check_sourcedata(self, _, expected_sourcedata):
        if False:
            i = 10
            return i + 15
        '\n        Assert that the sourcedata (from the patched functions - see\n        make_command) is correct.  Use this as a deferred callback.\n        '
        self.assertEqual(self.sourcedata, expected_sourcedata)
        return _