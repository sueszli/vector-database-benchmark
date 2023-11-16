import unittest
from coala_utils.ContextManagers import retrieve_stdout
from coalib.results.Result import Result
from coalib.results.result_actions.PrintMoreInfoAction import PrintMoreInfoAction
from coalib.settings.Section import Section

class PrintMoreInfoActionTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.uut = PrintMoreInfoAction()
        self.test_result = Result('origin', 'message', additional_info='A lot of additional information can be found here')

    def test_is_applicable(self):
        if False:
            print('Hello World!')
        with self.assertRaises(TypeError):
            self.uut.is_applicable(1, None, None)
        self.assertEqual(self.uut.is_applicable(Result('o', 'm'), None, None), 'There is no additional info.')
        self.assertTrue(self.uut.is_applicable(self.test_result, None, None))

    def test_apply(self):
        if False:
            print('Hello World!')
        with retrieve_stdout() as stdout:
            self.assertEqual(self.uut.apply_from_section(self.test_result, {}, {}, Section('name')), {})
            self.assertEqual(stdout.getvalue(), self.test_result.additional_info + '\n')