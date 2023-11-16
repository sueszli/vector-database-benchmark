from test.picardtestcase import PicardTestCase
from picard.util.remotecommands import RemoteCommands

class TestParsingFilesWithCommands(PicardTestCase):
    TEST_FILE = 'test/data/test-command-file-1.txt'

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.result = []
        RemoteCommands.set_quit(False)
        RemoteCommands.get_commands_from_file(self.TEST_FILE)
        while not RemoteCommands.command_queue.empty():
            (cmd, arg) = RemoteCommands.command_queue.get()
            self.result.append(f'{cmd} {arg}')
            RemoteCommands.command_queue.task_done()

    def test_no_argument_command(self):
        if False:
            while True:
                i = 10
        self.assertIn('CLUSTER ', self.result)

    def test_no_argument_command_stripped_correctly(self):
        if False:
            i = 10
            return i + 15
        self.assertIn('FINGERPRINT ', self.result)

    def test_single_argument_command(self):
        if False:
            return 10
        self.assertIn('LOAD file3.mp3', self.result)

    def test_multiple_arguments_command(self):
        if False:
            while True:
                i = 10
        self.assertIn('LOAD file1.mp3', self.result)
        self.assertIn('LOAD file2.mp3', self.result)

    def test_from_file_command_parsed(self):
        if False:
            print('Hello World!')
        self.assertNotIn('FROM_FILE command_file.txt', self.result)
        self.assertNotIn('FROM_FILE test/data/test-command-file-1.txt', self.result)
        self.assertNotIn('FROM_FILE test/data/test-command-file-2.txt', self.result)

    def test_noting_added_after_quit(self):
        if False:
            return 10
        self.assertNotIn('LOOKUP clustered', self.result)

    def test_empty_lines(self):
        if False:
            return 10
        self.assertNotIn(' ', self.result)
        self.assertNotIn('', self.result)
        self.assertEqual(len(self.result), 7)

    def test_commented_lines(self):
        if False:
            print('Hello World!')
        self.assertNotIn('#commented command', self.result)