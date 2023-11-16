from borb.license.persistent_random_user_id import PersistentRandomUserID
from borb.pdf import Document
from borb.pdf import PDF
from borb.pdf import Page
from borb.pdf import PageLayout
from borb.pdf import Paragraph
from borb.pdf import SingleColumnLayout
from tests.test_case import TestCase

class TestSendUsageStatistics(TestCase):
    """
    These tests check the anonymous usage statistics data mechanism
    """

    def test_send_usage_statistics(self):
        if False:
            for i in range(10):
                print('nop')

        def _get_user_id() -> str:
            if False:
                return 10
            return 'developer-user-id-jsc'
        prev_user_id_function = PersistentRandomUserID.get
        PersistentRandomUserID.get = _get_user_id
        d: Document = Document()
        p: Page = Page()
        d.add_page(p)
        l: PageLayout = SingleColumnLayout(p)
        for _ in range(0, 32):
            l.add(Paragraph('Hello World'))
        with open(self.get_first_output_file(), 'wb') as out_file_handle:
            PDF.dumps(out_file_handle, d)
        PersistentRandomUserID.get = prev_user_id_function