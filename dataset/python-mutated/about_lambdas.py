from runner.koan import *

class AboutLambdas(Koan):

    def test_lambdas_can_be_assigned_to_variables_and_called_explicitly(self):
        if False:
            i = 10
            return i + 15
        add_one = lambda n: n + 1
        self.assertEqual(__, add_one(10))

    def make_order(self, order):
        if False:
            while True:
                i = 10
        return lambda qty: str(qty) + ' ' + order + 's'

    def test_accessing_lambda_via_assignment(self):
        if False:
            for i in range(10):
                print('nop')
        sausages = self.make_order('sausage')
        eggs = self.make_order('egg')
        self.assertEqual(__, sausages(3))
        self.assertEqual(__, eggs(2))

    def test_accessing_lambda_without_assignment(self):
        if False:
            print('Hello World!')
        self.assertEqual(__, self.make_order('spam')(39823))