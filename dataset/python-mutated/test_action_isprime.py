from st2tests.base import BaseActionTestCase
from pythonactions.isprime import PrimeCheckerAction

class PrimeCheckerActionTestCase(BaseActionTestCase):
    action_cls = PrimeCheckerAction

    def test_run(self):
        if False:
            for i in range(10):
                print('nop')
        action = self.get_action_instance()
        result = action.run(value=1)
        self.assertFalse(result)
        result = action.run(value=3)
        self.assertTrue(result)