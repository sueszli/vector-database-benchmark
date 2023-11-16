"""Unit test utilities for gtest_xml_output"""
import re
from xml.dom import minidom, Node
import gtest_test_utils
GTEST_DEFAULT_OUTPUT_FILE = 'test_detail.xml'

class GTestXMLTestCase(gtest_test_utils.TestCase):
    """
  Base class for tests of Google Test's XML output functionality.
  """

    def AssertEquivalentNodes(self, expected_node, actual_node):
        if False:
            while True:
                i = 10
        '\n    Asserts that actual_node (a DOM node object) is equivalent to\n    expected_node (another DOM node object), in that either both of\n    them are CDATA nodes and have the same value, or both are DOM\n    elements and actual_node meets all of the following conditions:\n\n    *  It has the same tag name as expected_node.\n    *  It has the same set of attributes as expected_node, each with\n       the same value as the corresponding attribute of expected_node.\n       Exceptions are any attribute named "time", which needs only be\n       convertible to a floating-point number and any attribute named\n       "type_param" which only has to be non-empty.\n    *  It has an equivalent set of child nodes (including elements and\n       CDATA sections) as expected_node.  Note that we ignore the\n       order of the children as they are not guaranteed to be in any\n       particular order.\n    '
        if expected_node.nodeType == Node.CDATA_SECTION_NODE:
            self.assertEquals(Node.CDATA_SECTION_NODE, actual_node.nodeType)
            self.assertEquals(expected_node.nodeValue, actual_node.nodeValue)
            return
        self.assertEquals(Node.ELEMENT_NODE, actual_node.nodeType)
        self.assertEquals(Node.ELEMENT_NODE, expected_node.nodeType)
        self.assertEquals(expected_node.tagName, actual_node.tagName)
        expected_attributes = expected_node.attributes
        actual_attributes = actual_node.attributes
        self.assertEquals(expected_attributes.length, actual_attributes.length, 'attribute numbers differ in element %s:\nExpected: %r\nActual: %r' % (actual_node.tagName, expected_attributes.keys(), actual_attributes.keys()))
        for i in range(expected_attributes.length):
            expected_attr = expected_attributes.item(i)
            actual_attr = actual_attributes.get(expected_attr.name)
            self.assert_(actual_attr is not None, 'expected attribute %s not found in element %s' % (expected_attr.name, actual_node.tagName))
            self.assertEquals(expected_attr.value, actual_attr.value, ' values of attribute %s in element %s differ: %s vs %s' % (expected_attr.name, actual_node.tagName, expected_attr.value, actual_attr.value))
        expected_children = self._GetChildren(expected_node)
        actual_children = self._GetChildren(actual_node)
        self.assertEquals(len(expected_children), len(actual_children), 'number of child elements differ in element ' + actual_node.tagName)
        for (child_id, child) in expected_children.items():
            self.assert_(child_id in actual_children, '<%s> is not in <%s> (in element %s)' % (child_id, actual_children, actual_node.tagName))
            self.AssertEquivalentNodes(child, actual_children[child_id])
    identifying_attribute = {'testsuites': 'name', 'testsuite': 'name', 'testcase': 'name', 'failure': 'message', 'property': 'name'}

    def _GetChildren(self, element):
        if False:
            return 10
        '\n    Fetches all of the child nodes of element, a DOM Element object.\n    Returns them as the values of a dictionary keyed by the IDs of the\n    children.  For <testsuites>, <testsuite>, <testcase>, and <property>\n    elements, the ID is the value of their "name" attribute; for <failure>\n    elements, it is the value of the "message" attribute; for <properties>\n    elements, it is the value of their parent\'s "name" attribute plus the\n    literal string "properties"; CDATA sections and non-whitespace\n    text nodes are concatenated into a single CDATA section with ID\n    "detail".  An exception is raised if any element other than the above\n    four is encountered, if two child elements with the same identifying\n    attributes are encountered, or if any other type of node is encountered.\n    '
        children = {}
        for child in element.childNodes:
            if child.nodeType == Node.ELEMENT_NODE:
                if child.tagName == 'properties':
                    self.assert_(child.parentNode is not None, 'Encountered <properties> element without a parent')
                    child_id = child.parentNode.getAttribute('name') + '-properties'
                else:
                    self.assert_(child.tagName in self.identifying_attribute, 'Encountered unknown element <%s>' % child.tagName)
                    child_id = child.getAttribute(self.identifying_attribute[child.tagName])
                self.assert_(child_id not in children)
                children[child_id] = child
            elif child.nodeType in [Node.TEXT_NODE, Node.CDATA_SECTION_NODE]:
                if 'detail' not in children:
                    if child.nodeType == Node.CDATA_SECTION_NODE or not child.nodeValue.isspace():
                        children['detail'] = child.ownerDocument.createCDATASection(child.nodeValue)
                else:
                    children['detail'].nodeValue += child.nodeValue
            else:
                self.fail('Encountered unexpected node type %d' % child.nodeType)
        return children

    def NormalizeXml(self, element):
        if False:
            print('Hello World!')
        '\n    Normalizes Google Test\'s XML output to eliminate references to transient\n    information that may change from run to run.\n\n    *  The "time" attribute of <testsuites>, <testsuite> and <testcase>\n       elements is replaced with a single asterisk, if it contains\n       only digit characters.\n    *  The "timestamp" attribute of <testsuites> elements is replaced with a\n       single asterisk, if it contains a valid ISO8601 datetime value.\n    *  The "type_param" attribute of <testcase> elements is replaced with a\n       single asterisk (if it sn non-empty) as it is the type name returned\n       by the compiler and is platform dependent.\n    *  The line info reported in the first line of the "message"\n       attribute and CDATA section of <failure> elements is replaced with the\n       file\'s basename and a single asterisk for the line number.\n    *  The directory names in file paths are removed.\n    *  The stack traces are removed.\n    '
        if element.tagName == 'testsuites':
            timestamp = element.getAttributeNode('timestamp')
            timestamp.value = re.sub('^\\d{4}-\\d\\d-\\d\\dT\\d\\d:\\d\\d:\\d\\d$', '*', timestamp.value)
        if element.tagName in ('testsuites', 'testsuite', 'testcase'):
            time = element.getAttributeNode('time')
            time.value = re.sub('^\\d+(\\.\\d+)?$', '*', time.value)
            type_param = element.getAttributeNode('type_param')
            if type_param and type_param.value:
                type_param.value = '*'
        elif element.tagName == 'failure':
            source_line_pat = '^.*[/\\\\](.*:)\\d+\\n'
            message = element.getAttributeNode('message')
            message.value = re.sub(source_line_pat, '\\1*\n', message.value)
            for child in element.childNodes:
                if child.nodeType == Node.CDATA_SECTION_NODE:
                    cdata = re.sub(source_line_pat, '\\1*\n', child.nodeValue)
                    child.nodeValue = re.sub('Stack trace:\\n(.|\\n)*', 'Stack trace:\n*', cdata)
        for child in element.childNodes:
            if child.nodeType == Node.ELEMENT_NODE:
                self.NormalizeXml(child)