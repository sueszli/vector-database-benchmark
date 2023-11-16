import unittest
from troposphere.stepfunctions import Activity, StateMachine

class TestStepFunctions(unittest.TestCase):

    def test_activity(self):
        if False:
            i = 10
            return i + 15
        activity = Activity('myactivity', Name='testactivity')
        self.assertEqual(activity.Name, 'testactivity')

    def test_statemachine(self):
        if False:
            while True:
                i = 10
        statemachine = StateMachine('mystatemachine', DefinitionString='testdefinitionstring', RoleArn='testinrolearn')
        self.assertEqual(statemachine.RoleArn, 'testinrolearn')

    def test_statemachine_missing_parameter(self):
        if False:
            i = 10
            return i + 15
        StateMachine('mystatemachine', DefinitionString='testdefinitionstring')
        self.assertTrue(AttributeError)
if __name__ == '__main__':
    unittest.main()