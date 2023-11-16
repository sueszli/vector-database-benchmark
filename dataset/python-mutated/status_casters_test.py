"""Tests and examples for status_casters.h."""
from absl.testing import absltest
from xla.python import status_casters_ext

class StatusCastersTest(absltest.TestCase):

    def test_status_wrappers(self):
        if False:
            print('Hello World!')
        self.assertIsNone(status_casters_ext.my_lambda())
        self.assertIsNone(status_casters_ext.my_lambda2())
        self.assertIsNone(status_casters_ext.MyClass().my_method(1, 2))
        self.assertIsNone(status_casters_ext.MyClass().my_method_const(1, 2))

    def test_status_or_wrappers(self):
        if False:
            print('Hello World!')
        self.assertEqual(status_casters_ext.my_lambda_statusor(), 1)
        self.assertEqual(status_casters_ext.status_or_identity(2), 2)
        self.assertEqual(status_casters_ext.MyClass().my_method_status_or(1, 2), 3)
        self.assertEqual(status_casters_ext.MyClass().my_method_status_or_const(1, 2), 3)
if __name__ == '__main__':
    absltest.main()