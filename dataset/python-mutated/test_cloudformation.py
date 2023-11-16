import unittest
from troposphere import Ref
from troposphere.cloudformation import WaitCondition, WaitConditionHandle
from troposphere.policies import CreationPolicy, ResourceSignal

class TestWaitCondition(unittest.TestCase):

    def test_CreationPolicy(self):
        if False:
            i = 10
            return i + 15
        w = WaitCondition('mycondition', CreationPolicy=CreationPolicy(ResourceSignal=ResourceSignal(Timeout='PT15M')))
        w.validate()

    def test_CreationPolicyWithProps(self):
        if False:
            i = 10
            return i + 15
        w = WaitCondition('mycondition', Count=10, CreationPolicy=CreationPolicy(ResourceSignal=ResourceSignal(Timeout='PT15M')))
        with self.assertRaises(ValueError):
            w.validate()

    def test_RequiredProps(self):
        if False:
            while True:
                i = 10
        handle = WaitConditionHandle('myWaitHandle')
        w = WaitCondition('mycondition', Handle=Ref(handle), Timeout='300')
        w.validate()
if __name__ == '__main__':
    unittest.main()