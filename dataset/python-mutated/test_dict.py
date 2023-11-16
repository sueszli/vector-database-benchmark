import unittest
from troposphere import Ref, Tags, Template, ecs, iam

class TestDict(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.d = {'Cpu': 1, 'Environment': [{'Name': 'REGISTRY_STORAGE', 'Value': 's3'}, {'Name': 'REGISTRY_STORAGE_S3_REGION', 'Value': 'eu-west-1'}], 'Essential': True, 'Image': 'registry:2', 'Memory': 500, 'Name': 'registry', 'PortMappings': [{'ContainerPort': 5000, 'HostPort': 5000}, {'ContainerPort': 5001, 'HostPort': 5001}], 'Links': ['containerA', 'containerB']}

    def test_valid_data(self):
        if False:
            print('Hello World!')
        t = Template()
        cd = ecs.ContainerDefinition.from_dict('mycontainer', self.d)
        self.assertEqual(cd.Links[0], 'containerA')
        td = ecs.TaskDefinition('taskdef', ContainerDefinitions=[cd], Volumes=[ecs.Volume(Name='myvol')], TaskRoleArn=Ref(iam.Role('myecsrole')))
        t.add_resource(td)
        t.to_json()

    def test_invalid_toplevel_property(self):
        if False:
            return 10
        self.d['BlahInvalid'] = 'Invalid'
        with self.assertRaises(AttributeError):
            ecs.ContainerDefinition.from_dict('mycontainer', self.d)

    def test_invalid_sub_property(self):
        if False:
            return 10
        self.d['Environment'][0]['BlahInvalid'] = 'Invalid'
        with self.assertRaises(AttributeError):
            ecs.ContainerDefinition.from_dict('mycontainer', self.d)

    def test_toplevel_helper_fn(self):
        if False:
            return 10
        self.d['Cpu'] = Ref('MyCPU')
        cd = ecs.ContainerDefinition.from_dict('mycontainer', self.d)
        self.assertEqual(cd.Cpu.data, {'Ref': 'MyCPU'})

    def test_sub_property_helper_fn(self):
        if False:
            while True:
                i = 10
        self.d['Environment'][0]['Value'] = Ref('RegistryStorage')
        cd = ecs.ContainerDefinition.from_dict('mycontainer', self.d)
        self.assertEqual(cd.Environment[0].Value.data, {'Ref': 'RegistryStorage'})

    def test_invalid_subproperty_definition(self):
        if False:
            i = 10
            return i + 15
        self.d['Environment'][0] = 'BadValue'
        with self.assertRaises(ValueError):
            ecs.ContainerDefinition.from_dict('mycontainer', self.d)

    def test_tags_from_dict(self):
        if False:
            for i in range(10):
                print('nop')

        def key(k):
            if False:
                while True:
                    i = 10
            return (k['Key'], k['Value'])
        d = {'key1': 'value1', 'key2': 'value2'}
        expected = [{'Key': k, 'Value': v} for (k, v) in d.items()]
        tags = Tags.from_dict(**d)
        self.assertEqual(sorted(expected, key=key), sorted(tags.tags, key=key))
if __name__ == '__main__':
    unittest.main()