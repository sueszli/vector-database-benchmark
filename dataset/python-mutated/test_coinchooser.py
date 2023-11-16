from electrum.coinchooser import CoinChooserPrivacy
from electrum.util import NotEnoughFunds
from . import ElectrumTestCase

class TestCoinChooser(ElectrumTestCase):

    def test_bucket_candidates_with_empty_buckets(self):
        if False:
            print('Hello World!')

        def sufficient_funds(buckets, *, bucket_value_sum):
            if False:
                print('Hello World!')
            return True
        coin_chooser = CoinChooserPrivacy(enable_output_value_rounding=False)
        self.assertEqual([[]], coin_chooser.bucket_candidates_any([], sufficient_funds))
        self.assertEqual([[]], coin_chooser.bucket_candidates_prefer_confirmed([], sufficient_funds))

        def sufficient_funds(buckets, *, bucket_value_sum):
            if False:
                while True:
                    i = 10
            return False
        with self.assertRaises(NotEnoughFunds):
            coin_chooser.bucket_candidates_any([], sufficient_funds)
        with self.assertRaises(NotEnoughFunds):
            coin_chooser.bucket_candidates_prefer_confirmed([], sufficient_funds)