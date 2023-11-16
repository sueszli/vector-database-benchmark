import unittest
import tap_facebook.__init__ as tap_facebook

class TestAttributionWindow(unittest.TestCase):
    """
        Test case to verify that proper error message is raise
        when user enters attribution window other than 1, 7 and 28
    """

    def test_invalid_attribution_window(self):
        if False:
            return 10
        error_message = None
        tap_facebook.CONFIG = {'start_date': '2019-01-01T00:00:00Z', 'account_id': 'test_account_id', 'access_token': 'test_access_token', 'insights_buffer_days': 30}
        try:
            tap_facebook.AdsInsights('test', 'test', 'test', None, {}, {})
        except Exception as e:
            error_message = str(e)
        self.assertEquals(error_message, 'The attribution window must be 1, 7 or 28.')