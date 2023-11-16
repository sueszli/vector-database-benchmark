from sentry.testutils.cases import AcceptanceTestCase
from sentry.testutils.helpers.response import close_streaming_response

class TestChartRenderer(AcceptanceTestCase):

    def test_debug_renders(self):
        if False:
            print('Hello World!')
        options = {'chart-rendering.enabled': True, 'system.url-prefix': self.browser.live_server_url}
        with self.options(options):
            self.browser.get('debug/chart-renderer/')
        images = self.browser.elements(selector='img')
        assert len(images) > 0
        for image in images:
            src = image.get_attribute('src')
            resp = self.client.get(src)
            assert resp.status_code == 200
            assert close_streaming_response(resp)[:4] == b'\x89PNG'