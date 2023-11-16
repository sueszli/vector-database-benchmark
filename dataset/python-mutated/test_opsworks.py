import unittest
from troposphere import Template
from troposphere.opsworks import Stack

class TestOpsWorksStack(unittest.TestCase):

    def test_nosubnet(self):
        if False:
            return 10
        stack = Stack('mystack', VpcId='myvpcid')
        with self.assertRaises(ValueError):
            stack.validate()

    def test_stack(self):
        if False:
            for i in range(10):
                print('nop')
        stack = Stack('mystack', VpcId='myvpcid', DefaultSubnetId='subnetid')
        self.assertIsNone(stack.validate())

    def test_no_required(self):
        if False:
            for i in range(10):
                print('nop')
        stack = Stack('mystack')
        t = Template()
        t.add_resource(stack)
        with self.assertRaises(ValueError):
            t.to_json()

    def test_required(self):
        if False:
            return 10
        stack = Stack('mystack', DefaultInstanceProfileArn='instancearn', Name='myopsworksname', ServiceRoleArn='arn')
        t = Template()
        t.add_resource(stack)
        t.to_json()

    def test_custom_json(self):
        if False:
            while True:
                i = 10
        stack = Stack('mystack', DefaultInstanceProfileArn='instancearn', Name='myopsworksname', ServiceRoleArn='arn')
        t = Template()
        stack.CustomJson = {'foo': 'bar'}
        t.add_resource(stack)
        t.to_json()
        t = Template()
        stack.CustomJson = '{"foo": "bar"}'
        t.add_resource(stack)
        t.to_json()
        with self.assertRaises(TypeError):
            stack.CustomJson = True
if __name__ == '__main__':
    unittest.main()