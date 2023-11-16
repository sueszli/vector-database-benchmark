import unittest
from io import StringIO
from unittest.mock import patch
from .. import _report_results
from ..model import ClassModel

class InitTest(unittest.TestCase):

    @patch('sys.stdout', new_callable=StringIO)
    def test_report_results(self, mock_stdout: StringIO) -> None:
        if False:
            print('Hello World!')
        models = {'generator_one': [ClassModel('Class1', 'Annotation1'), ClassModel('Class2', 'Annotation1')], 'generator_two': [ClassModel('Class2', 'Annotation2'), ClassModel('Class3', 'Annotation2')]}
        _report_results(models, None)
        self.assertEqual(mock_stdout.getvalue(), '\n'.join(['class Class1(Annotation1): ...', 'class Class2(Annotation1): ...', 'class Class2(Annotation2): ...', 'class Class3(Annotation2): ...', '']))