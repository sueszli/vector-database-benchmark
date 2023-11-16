"""
Console module tests
"""
import contextlib
import io
import os
import tempfile
import unittest
from txtai.console import Console
from txtai.embeddings import Embeddings
APPLICATION = '\npath: %s\n\nworkflow:\n  test:\n     tasks:\n       - task: console\n'

class TestConsole(unittest.TestCase):
    """
    Console tests.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        '\n        Initialize test data.\n        '
        cls.data = ['US tops 5 million confirmed virus cases', "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg", 'Beijing mobilises invasion craft along coast as Taiwan tensions escalate', 'The National Park Service warns against sacrificing slower friends in a bear attack', 'Maine man wins $1M from $25 lottery ticket', 'Make huge profits without work, earn up to $100,000 a day']
        cls.embeddings = Embeddings({'path': 'sentence-transformers/nli-mpnet-base-v2', 'content': True})
        cls.embeddings.index([(uid, text, None) for (uid, text) in enumerate(cls.data)])
        cls.apppath = os.path.join(tempfile.gettempdir(), 'console.yml')
        cls.embedpath = os.path.join(tempfile.gettempdir(), 'embeddings.console')
        with open(cls.apppath, 'w', encoding='utf-8') as out:
            out.write(APPLICATION % cls.embedpath)
        cls.embeddings.save(cls.embedpath)
        cls.embeddings.save(f'{cls.embedpath}.tar.gz')
        cls.console = Console(cls.embedpath)

    def testApplication(self):
        if False:
            while True:
                i = 10
        '\n        Test application\n        '
        self.assertNotIn('Traceback', self.command(f'.load {self.apppath}'))
        self.assertIn('1', self.command('.limit 1'))
        self.assertIn('Maine man wins', self.command('feel good story'))

    def testConfig(self):
        if False:
            print('Hello World!')
        '\n        Test .config command\n        '
        self.assertIn('tasks', self.command('.config'))

    def testEmbeddings(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test embeddings index\n        '
        self.assertNotIn('Traceback', self.command(f'.load {self.embedpath}.tar.gz'))
        self.assertNotIn('Traceback', self.command(f'.load {self.embedpath}'))
        self.assertIn('1', self.command('.limit 1'))
        self.assertIn('Maine man wins', self.command('feel good story'))

    def testEmbeddingsNoDatabase(self):
        if False:
            i = 10
            return i + 15
        '\n        Test embeddings with no database/content\n        '
        console = Console()
        embeddings = Embeddings({'path': 'sentence-transformers/nli-mpnet-base-v2'})
        embeddings.index([(uid, text, None) for (uid, text) in enumerate(self.data)])
        console.app = embeddings
        self.assertIn('4', self.command('feel good story', console))

    def testEmpty(self):
        if False:
            while True:
                i = 10
        '\n        Test empty console instance\n        '
        console = Console()
        self.assertIn('AttributeError', self.command('search', console))

    def testHighlight(self):
        if False:
            return 10
        '\n        Test .highlight command\n        '
        self.assertIn('highlight', self.command('.highlight'))
        self.assertIn('wins', self.command('feel good story'))
        self.assertIn('Taiwan', self.command('asia'))

    def testPreloop(self):
        if False:
            return 10
        '\n        Test preloop\n        '
        self.assertIn('txtai console', self.preloop())

    def testWorkflow(self):
        if False:
            print('Hello World!')
        '\n        Test .workflow command\n        '
        self.command(f'.load {self.apppath}')
        self.assertIn('echo', self.command('.workflow test echo'))

    def command(self, command, console=None):
        if False:
            while True:
                i = 10
        '\n        Runs a console command.\n\n        Args:\n            command: command to run\n            console: console instance, defaults to self.console\n\n        Returns:\n            command output\n        '
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            if not console:
                console = self.console
            console.onecmd(command)
        return output.getvalue()

    def preloop(self):
        if False:
            print('Hello World!')
        '\n        Runs console.preloop and redirects stdout.\n\n        Returns:\n            preloop output\n        '
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.console.preloop()
        return output.getvalue()