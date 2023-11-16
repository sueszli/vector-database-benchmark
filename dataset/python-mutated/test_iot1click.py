import unittest
from troposphere import Template
from troposphere.iot1click import PlacementTemplate, Project

class TestPlacementTemplate(unittest.TestCase):

    def test_placement_template(self):
        if False:
            print('Hello World!')
        template = Template()
        template.add_resource(Project('BasicProject', ProjectName='project', Description='description', PlacementTemplate=PlacementTemplate(DefaultAttributes={'Attribute': 'Value', 'Foo': 'Bar'}, DeviceTemplates={'testbutton': {'DeviceType': 'button'}})))
        template.to_json()