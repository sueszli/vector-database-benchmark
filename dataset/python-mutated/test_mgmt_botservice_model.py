from azure.mgmt.botservice.models import Site

class TestMgmtBotServiceModel:

    def test_model_site(self):
        if False:
            for i in range(10):
                print('nop')
        Site(is_v1_enabled=True, is_v3_enabled=True, site_name='xxx', is_enabled=True, is_webchat_preview_enabled=True, is_secure_site_enabled=True)