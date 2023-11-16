from unittest import TestCase, expectedFailure

class FailureTestCase(TestCase):

    def test_sample(self):
        if False:
            return 10
        self.assertEqual(0, 1)

class ErrorTestCase(TestCase):

    def test_sample(self):
        if False:
            print('Hello World!')
        raise Exception('test')

class ExpectedFailureTestCase(TestCase):

    @expectedFailure
    def test_sample(self):
        if False:
            print('Hello World!')
        self.assertEqual(0, 1)

class UnexpectedSuccessTestCase(TestCase):

    @expectedFailure
    def test_sample(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(1, 1)