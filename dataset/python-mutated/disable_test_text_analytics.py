from azure.cognitiveservices.language.textanalytics import TextAnalyticsClient
from devtools_testutils import ResourceGroupPreparer
from devtools_testutils.cognitiveservices_testcase import CognitiveServiceTest, CognitiveServicesAccountPreparer

class TextAnalyticsTest(CognitiveServiceTest):

    @ResourceGroupPreparer()
    @CognitiveServicesAccountPreparer(name_prefix='pycog', legacy=True)
    def test_detect_language(self, resource_group, location, cognitiveservices_account, cognitiveservices_account_key):
        if False:
            return 10
        text_analytics = TextAnalyticsClient(cognitiveservices_account, cognitiveservices_account_key)
        response = text_analytics.detect_language(documents=[{'id': 1, 'text': 'I had a wonderful experience! The rooms were wonderful and the staff was helpful.'}])
        self.assertEqual(response.documents[0].detected_languages[0].name, 'English')