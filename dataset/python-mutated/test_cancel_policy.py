from unittest import TestCase
from zipline.finance.cancel_policy import NeverCancel, EODCancel
from zipline.gens.sim_engine import BAR, SESSION_END

class CancelPolicyTestCase(TestCase):

    def test_eod_cancel(self):
        if False:
            print('Hello World!')
        cancel_policy = EODCancel()
        self.assertTrue(cancel_policy.should_cancel(SESSION_END))
        self.assertFalse(cancel_policy.should_cancel(BAR))

    def test_never_cancel(self):
        if False:
            i = 10
            return i + 15
        cancel_policy = NeverCancel()
        self.assertFalse(cancel_policy.should_cancel(SESSION_END))
        self.assertFalse(cancel_policy.should_cancel(BAR))