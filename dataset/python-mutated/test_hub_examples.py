import unittest
from modelscope.hub.api import HubApi
from modelscope.utils.hub import create_model_if_not_exist
YOUR_ACCESS_TOKEN = 'token'

class HubExampleTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.api = HubApi()
        self.api.login(YOUR_ACCESS_TOKEN)

    @unittest.skip('to be used for local test only')
    def test_example_model_creation(self):
        if False:
            return 10
        model_name = 'cv_unet_person-image-cartoon_compound-models'
        model_chinese_name = '达摩卡通化模型'
        model_org = 'damo'
        model_id = '%s/%s' % (model_org, model_name)
        created = create_model_if_not_exist(self.api, model_id, model_chinese_name)
        if not created:
            print('!! NOT created since model already exists !!')
if __name__ == '__main__':
    unittest.main()