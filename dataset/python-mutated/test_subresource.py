from boto3.docs.subresource import SubResourceDocumenter
from tests.unit.docs import BaseDocsTest

class TestSubResourceDocumenter(BaseDocsTest):

    def test_document_sub_resources(self):
        if False:
            return 10
        sub_resource_documentor = SubResourceDocumenter(self.resource, self.root_services_path)
        sub_resource_documentor.document_sub_resources(self.doc_structure)
        self.assert_contains_lines_in_order(['-------------\nSub-resources\n-------------', 'Sub-resources are methods that create a new instance of a', " child resource. This resource's identifiers get passed", ' along to the child.', 'For more information about sub-resources refer to the '])
        self.assert_contains_lines_in_order(['Sample', '.. py:method:: MyService.ServiceResource.Sample(name)', '  Creates a Sample resource.::', "    sample = myservice.Sample('name')", '  :type name: string', "  :param name: The Sample's name identifier.", '  :rtype: :py:class:`MyService.Sample`', '  :returns: A Sample resource'], self.get_nested_service_contents('myservice', 'service-resource', 'Sample'))