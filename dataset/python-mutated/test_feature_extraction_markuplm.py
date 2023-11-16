import unittest
from transformers.testing_utils import require_bs4
from transformers.utils import is_bs4_available
from ...test_feature_extraction_common import FeatureExtractionSavingTestMixin
if is_bs4_available():
    from transformers import MarkupLMFeatureExtractor

class MarkupLMFeatureExtractionTester(unittest.TestCase):

    def __init__(self, parent):
        if False:
            i = 10
            return i + 15
        self.parent = parent

    def prepare_feat_extract_dict(self):
        if False:
            for i in range(10):
                print('nop')
        return {}

def get_html_strings():
    if False:
        for i in range(10):
            print('nop')
    html_string_1 = '<HTML>\n\n    <HEAD>\n    <TITLE>sample document</TITLE>\n    </HEAD>\n\n    <BODY BGCOLOR="FFFFFF">\n    <HR>\n    <a href="http://google.com">Goog</a>\n    <H1>This is one header</H1>\n    <H2>This is a another Header</H2>\n    <P>Travel from\n        <P>\n        <B>SFO to JFK</B>\n        <BR>\n        <B><I>on May 2, 2015 at 2:00 pm. For details go to confirm.com </I></B>\n        <HR>\n        <div style="color:#0000FF">\n            <h3>Traveler <b> name </b> is\n            <p> John Doe </p>\n        </div>'
    html_string_2 = '\n    <!DOCTYPE html>\n    <html>\n    <body>\n\n    <h1>My First Heading</h1>\n    <p>My first paragraph.</p>\n\n    </body>\n    </html>\n    '
    return [html_string_1, html_string_2]

@require_bs4
class MarkupLMFeatureExtractionTest(FeatureExtractionSavingTestMixin, unittest.TestCase):
    feature_extraction_class = MarkupLMFeatureExtractor if is_bs4_available() else None

    def setUp(self):
        if False:
            while True:
                i = 10
        self.feature_extract_tester = MarkupLMFeatureExtractionTester(self)

    @property
    def feat_extract_dict(self):
        if False:
            while True:
                i = 10
        return self.feature_extract_tester.prepare_feat_extract_dict()

    def test_call(self):
        if False:
            print('Hello World!')
        feature_extractor = self.feature_extraction_class()
        html_string = get_html_strings()[0]
        encoding = feature_extractor(html_string)
        expected_nodes = [['sample document', 'Goog', 'This is one header', 'This is a another Header', 'Travel from', 'SFO to JFK', 'on May 2, 2015 at 2:00 pm. For details go to confirm.com', 'Traveler', 'name', 'is', 'John Doe']]
        expected_xpaths = [['/html/head/title', '/html/body/a', '/html/body/h1', '/html/body/h2', '/html/body/p', '/html/body/p/p/b[1]', '/html/body/p/p/b[2]/i', '/html/body/p/p/div/h3', '/html/body/p/p/div/h3/b', '/html/body/p/p/div/h3', '/html/body/p/p/div/h3/p']]
        self.assertEqual(encoding.nodes, expected_nodes)
        self.assertEqual(encoding.xpaths, expected_xpaths)
        html_strings = get_html_strings()
        encoding = feature_extractor(html_strings)
        expected_nodes = expected_nodes + [['My First Heading', 'My first paragraph.']]
        expected_xpaths = expected_xpaths + [['/html/body/h1', '/html/body/p']]
        self.assertEqual(len(encoding.nodes), 2)
        self.assertEqual(len(encoding.xpaths), 2)
        self.assertEqual(encoding.nodes, expected_nodes)
        self.assertEqual(encoding.xpaths, expected_xpaths)