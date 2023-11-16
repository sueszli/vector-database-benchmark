from test.picardtestcase import PicardTestCase
from picard.ui.options.plugins import PluginsOptionsPage

class PluginsOptionsPageTest(PicardTestCase):

    def test_link_authors(self):
        if False:
            print('Hello World!')
        self.assertEqual('<a href="mailto:coyote@acme.com">Wile E. Coyote</a>, Road &lt;Runner&gt;', PluginsOptionsPage.link_authors('Wile E. Coyote <coyote@acme.com>, Road <Runner>'))