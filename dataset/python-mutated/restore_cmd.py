from typing import TextIO, Callable
from trashcli.lib.my_input import Input
from trashcli.lib.print_version import PrintVersionAction, PrintVersionArgs
from trashcli.restore.args import RunRestoreArgs
from trashcli.restore.file_system import RestoreReadFileSystem, RestoreWriteFileSystem, ReadCwd
from trashcli.restore.handler import HandlerImpl
from trashcli.restore.real_output import RealOutput
from trashcli.restore.restore_arg_parser import RestoreArgParser
from trashcli.restore.restorer import Restorer
from trashcli.restore.run_restore_action import RunRestoreAction, Handler
from trashcli.restore.trashed_files import TrashedFiles

class RestoreCmd(object):

    @staticmethod
    def make(stdout, stderr, exit, input, version, trashed_files, read_fs, write_fs, read_cwd):
        if False:
            for i in range(10):
                print('nop')
        restorer = Restorer(read_fs, write_fs)
        output = RealOutput(stdout, stderr, exit)
        handler = HandlerImpl(input, read_cwd, restorer, output)
        return RestoreCmd(stdout, version, trashed_files, read_cwd, handler)

    def __init__(self, stdout, version, trashed_files, read_cwd, handler):
        if False:
            while True:
                i = 10
        self.read_cwd = read_cwd
        self.parser = RestoreArgParser()
        self.run_restore_action = RunRestoreAction(handler, trashed_files)
        self.print_version_action = PrintVersionAction(stdout, version)

    def run(self, argv):
        if False:
            while True:
                i = 10
        args = self.parser.parse_restore_args(argv, self.read_cwd.getcwd_as_realpath())
        if isinstance(args, RunRestoreArgs):
            self.run_restore_action.run_action(args)
        elif isinstance(args, PrintVersionArgs):
            self.print_version_action.run_action(args)