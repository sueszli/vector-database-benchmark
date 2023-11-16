from __future__ import annotations
import sys
from ansible.module_utils.common.text.converters import to_text, to_bytes
from ansible.module_utils.connection import Connection, ConnectionError
try:
    from ncclient.xml_ import NCElement, new_ele, sub_ele
    HAS_NCCLIENT = True
except (ImportError, AttributeError):
    HAS_NCCLIENT = False
try:
    from lxml.etree import Element, fromstring, XMLSyntaxError
except ImportError:
    from xml.etree.ElementTree import Element, fromstring
    if sys.version_info < (2, 7):
        from xml.parsers.expat import ExpatError as XMLSyntaxError
    else:
        from xml.etree.ElementTree import ParseError as XMLSyntaxError
NS_MAP = {'nc': 'urn:ietf:params:xml:ns:netconf:base:1.0'}

def exec_rpc(module, *args, **kwargs):
    if False:
        print('Hello World!')
    connection = NetconfConnection(module._socket_path)
    return connection.execute_rpc(*args, **kwargs)

class NetconfConnection(Connection):

    def __init__(self, socket_path):
        if False:
            return 10
        super(NetconfConnection, self).__init__(socket_path)

    def __rpc__(self, name, *args, **kwargs):
        if False:
            print('Hello World!')
        'Executes the json-rpc and returns the output received\n           from remote device.\n           :name: rpc method to be executed over connection plugin that implements jsonrpc 2.0\n           :args: Ordered list of params passed as arguments to rpc method\n           :kwargs: Dict of valid key, value pairs passed as arguments to rpc method\n\n           For usage refer the respective connection plugin docs.\n        '
        self.check_rc = kwargs.pop('check_rc', True)
        self.ignore_warning = kwargs.pop('ignore_warning', True)
        response = self._exec_jsonrpc(name, *args, **kwargs)
        if 'error' in response:
            rpc_error = response['error'].get('data')
            return self.parse_rpc_error(to_bytes(rpc_error, errors='surrogate_then_replace'))
        return fromstring(to_bytes(response['result'], errors='surrogate_then_replace'))

    def parse_rpc_error(self, rpc_error):
        if False:
            print('Hello World!')
        if self.check_rc:
            try:
                error_root = fromstring(rpc_error)
                root = Element('root')
                root.append(error_root)
                error_list = root.findall('.//nc:rpc-error', NS_MAP)
                if not error_list:
                    raise ConnectionError(to_text(rpc_error, errors='surrogate_then_replace'))
                warnings = []
                for error in error_list:
                    message_ele = error.find('./nc:error-message', NS_MAP)
                    if message_ele is None:
                        message_ele = error.find('./nc:error-info', NS_MAP)
                    message = message_ele.text if message_ele is not None else None
                    severity = error.find('./nc:error-severity', NS_MAP).text
                    if severity == 'warning' and self.ignore_warning and (message is not None):
                        warnings.append(message)
                    else:
                        raise ConnectionError(to_text(rpc_error, errors='surrogate_then_replace'))
                return warnings
            except XMLSyntaxError:
                raise ConnectionError(rpc_error)

def transform_reply():
    if False:
        while True:
            i = 10
    return b'<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">\n    <xsl:output method="xml" indent="no"/>\n\n    <xsl:template match="/|comment()|processing-instruction()">\n        <xsl:copy>\n            <xsl:apply-templates/>\n        </xsl:copy>\n    </xsl:template>\n\n    <xsl:template match="*">\n        <xsl:element name="{local-name()}">\n            <xsl:apply-templates select="@*|node()"/>\n        </xsl:element>\n    </xsl:template>\n\n    <xsl:template match="@*">\n        <xsl:attribute name="{local-name()}">\n            <xsl:value-of select="."/>\n        </xsl:attribute>\n    </xsl:template>\n    </xsl:stylesheet>\n    '

def remove_namespaces(data):
    if False:
        print('Hello World!')
    if not HAS_NCCLIENT:
        raise ImportError('ncclient is required but does not appear to be installed.  It can be installed using `pip install ncclient`')
    return NCElement(data, transform_reply()).data_xml

def build_root_xml_node(tag):
    if False:
        i = 10
        return i + 15
    return new_ele(tag)

def build_child_xml_node(parent, tag, text=None, attrib=None):
    if False:
        i = 10
        return i + 15
    element = sub_ele(parent, tag)
    if text:
        element.text = to_text(text)
    if attrib:
        element.attrib.update(attrib)
    return element

def build_subtree(parent, path):
    if False:
        print('Hello World!')
    element = parent
    for field in path.split('/'):
        sub_element = build_child_xml_node(element, field)
        element = sub_element
    return element