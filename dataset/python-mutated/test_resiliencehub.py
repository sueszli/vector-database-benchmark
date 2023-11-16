import unittest
from troposphere.resiliencehub import FailurePolicy, ResiliencyPolicy

class TestResiliencyPolicy(unittest.TestCase):

    def test_ResiliencyPolicy(self):
        if False:
            for i in range(10):
                print('nop')
        ResiliencyPolicy('policy', Policy={'Hardware': FailurePolicy(RpoInSecs=10, RtoInSecs=10)}, PolicyName='foo', Tier='MissionCritical').to_dict()

    def test_invalid_policy_key(self):
        if False:
            return 10
        with self.assertRaises(ValueError):
            ResiliencyPolicy('policy', Policy={'Foo': FailurePolicy(RpoInSecs=10, RtoInSecs=10)}, PolicyName='foo', Tier='MissionCritical').to_dict()

    def test_invalid_policy_value(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            ResiliencyPolicy('policy', Policy={'Hardware': 10}, PolicyName='foo', Tier='MissionCritical').to_dict()

    def test_invalid_policy_tier(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            ResiliencyPolicy('policy', Policy={'Hardware': FailurePolicy(RpoInSecs=10, RtoInSecs=10)}, PolicyName='foo', Tier='foobar').to_dict()
if __name__ == '__main__':
    unittest.main()