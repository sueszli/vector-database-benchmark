import unittest
from troposphere import Output, Ref, Sub, Template, s3
from troposphere.cloudformation import WaitCondition
s3_bucket_yaml = "Description: S3 Bucket Example\nOutputs:\n  BucketName:\n    Description: Name of S3 bucket to hold website content\n    Value: !Ref 'S3Bucket'\nResources:\n  S3Bucket:\n    Properties:\n      AccessControl: PublicRead\n    Type: AWS::S3::Bucket\n"
cond_string = 'My\nspecial multiline\nHandle'
cond_normal = 'Resources:\n  MyWaitCondition:\n    Properties:\n      Handle: !Sub "My\\nspecial multiline\\nHandle"\n      Timeout: \'30\'\n    Type: AWS::CloudFormation::WaitCondition\n'
cond_long = 'Resources:\n  MyWaitCondition:\n    Properties:\n      Handle:\n        Fn::Sub: "My\\nspecial multiline\\nHandle"\n      Timeout: \'30\'\n    Type: AWS::CloudFormation::WaitCondition\n'
cond_clean = "Resources:\n  MyWaitCondition:\n    Properties:\n      Handle: !Sub |-\n        My\n        special multiline\n        Handle\n      Timeout: '30'\n    Type: AWS::CloudFormation::WaitCondition\n"

class TestYAML(unittest.TestCase):

    def test_s3_bucket(self):
        if False:
            while True:
                i = 10
        t = Template()
        t.set_description('S3 Bucket Example')
        s3bucket = t.add_resource(s3.Bucket('S3Bucket', AccessControl=s3.PublicRead))
        t.add_output(Output('BucketName', Value=Ref(s3bucket), Description='Name of S3 bucket to hold website content'))
        self.assertEqual(s3_bucket_yaml, t.to_yaml())

    def test_yaml_long_form(self):
        if False:
            return 10
        t = Template()
        t.add_resource(WaitCondition('MyWaitCondition', Timeout=30, Handle=Sub(cond_string)))
        self.assertEqual(cond_normal, t.to_yaml())
        self.assertEqual(cond_long, t.to_yaml(long_form=True))
        self.assertEqual(cond_long, t.to_yaml(False, True))
        self.assertEqual(cond_clean, t.to_yaml(clean_up=True))
        self.assertEqual(cond_clean, t.to_yaml(True))