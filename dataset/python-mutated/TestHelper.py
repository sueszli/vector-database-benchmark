import os
from stat import S_IREAD, S_IWRITE, S_IEXEC
from robot.api import logger

class TestHelper:

    def set_read_only(self, path):
        if False:
            print('Hello World!')
        os.chmod(path, S_IREAD)

    def set_read_write(self, path):
        if False:
            print('Hello World!')
        os.chmod(path, S_IREAD | S_IWRITE)

    def set_read_write_execute(self, path):
        if False:
            return 10
        os.chmod(path, S_IREAD | S_IWRITE | S_IEXEC)

    def remove_permissions(self, path):
        if False:
            return 10
        os.chmod(path, 0)

    def file_should_have_correct_line_separators(self, output, sep=os.linesep):
        if False:
            i = 10
            return i + 15
        if os.path.isfile(output):
            with open(output, 'rb') as infile:
                output = infile.read().decode('UTF-8')
        if sep not in output:
            self._wrong_separators('Output has no %r separators' % sep, output)
        extra_r = output.replace(sep, '').count('\r')
        extra_n = output.replace(sep, '').count('\n')
        if extra_r or extra_n:
            self._wrong_separators('Output has %d extra \\r and %d extra \\n' % (extra_r, extra_n), output)

    def _wrong_separators(self, message, output):
        if False:
            return 10
        logger.info(repr(output).replace('\\n', '\\n\n'))
        failure = AssertionError(message)
        failure.ROBOT_CONTINUE_ON_FAILURE = True
        raise failure