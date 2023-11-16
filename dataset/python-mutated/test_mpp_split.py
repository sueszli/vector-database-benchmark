import random
import electrum.mpp_split as mpp_split
from electrum.lnutil import NoPathFound
from . import ElectrumTestCase
PART_PENALTY = mpp_split.PART_PENALTY

class TestMppSplit(ElectrumTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        random.seed(0)
        self.channels_with_funds = {(b'0', b'0'): 1000000000, (b'1', b'1'): 500000000, (b'2', b'0'): 302000000, (b'3', b'2'): 101000000}

    def tearDown(self):
        if False:
            print('Hello World!')
        super().tearDown()
        mpp_split.PART_PENALTY = PART_PENALTY

    def test_suggest_splits(self):
        if False:
            return 10
        with self.subTest(msg='do a payment with the maximal amount spendable over a single channel'):
            splits = mpp_split.suggest_splits(1000000000, self.channels_with_funds, exclude_single_part_payments=True)
            self.assertEqual({(b'0', b'0'): [671020676], (b'1', b'1'): [328979324], (b'2', b'0'): [], (b'3', b'2'): []}, splits[0].config)
        with self.subTest(msg='payment amount that does not require to be split'):
            splits = mpp_split.suggest_splits(50000000, self.channels_with_funds, exclude_single_part_payments=False)
            self.assertEqual({(b'0', b'0'): [50000000]}, splits[0].config)
            self.assertEqual({(b'1', b'1'): [50000000]}, splits[1].config)
            self.assertEqual({(b'2', b'0'): [50000000]}, splits[2].config)
            self.assertEqual({(b'3', b'2'): [50000000]}, splits[3].config)
            self.assertEqual(2, splits[4].config.number_parts())
        with self.subTest(msg='do a payment with a larger amount than what is supported by a single channel'):
            splits = mpp_split.suggest_splits(1100000000, self.channels_with_funds, exclude_single_part_payments=False)
            self.assertEqual(2, splits[0].config.number_parts())
        with self.subTest(msg='do a payment with the maximal amount spendable over all channels'):
            splits = mpp_split.suggest_splits(sum(self.channels_with_funds.values()), self.channels_with_funds, exclude_single_part_payments=True)
            self.assertEqual({(b'0', b'0'): [1000000000], (b'1', b'1'): [500000000], (b'2', b'0'): [302000000], (b'3', b'2'): [101000000]}, splits[0].config)
        with self.subTest(msg='do a payment with the amount supported by all channels'):
            splits = mpp_split.suggest_splits(101000000, self.channels_with_funds, exclude_single_part_payments=False)
            for split in splits[:3]:
                self.assertEqual(1, split.config.number_nonzero_channels())
            self.assertEqual(2, splits[4].config.number_parts())

    def test_send_to_single_node(self):
        if False:
            print('Hello World!')
        splits = mpp_split.suggest_splits(1000000000, self.channels_with_funds, exclude_single_part_payments=False, exclude_multinode_payments=True)
        for split in splits:
            assert split.config.number_nonzero_nodes() == 1

    def test_saturation(self):
        if False:
            i = 10
            return i + 15
        'Split configurations which spend the full amount in a channel should be avoided.'
        channels_with_funds = {(b'0', b'0'): 159799733076, (b'1', b'1'): 499986152000}
        splits = mpp_split.suggest_splits(600000000000, channels_with_funds, exclude_single_part_payments=True)
        uses_full_amount = False
        for (c, a) in splits[0].config.items():
            if a == channels_with_funds[c]:
                uses_full_amount |= True
        self.assertFalse(uses_full_amount)

    def test_payment_below_min_part_size(self):
        if False:
            return 10
        amount = mpp_split.MIN_PART_SIZE_MSAT // 2
        splits = mpp_split.suggest_splits(amount, self.channels_with_funds, exclude_single_part_payments=False)
        self.assertEqual(4, len(splits))

    def test_suggest_part_penalty(self):
        if False:
            i = 10
            return i + 15
        'Test is mainly for documentation purposes.\n        Decreasing the part penalty from 1.0 towards 0.0 leads to an increase\n        in the number of parts a payment is split. A configuration which has\n        about equally distributed amounts will result.'
        with self.subTest(msg='split payments with intermediate part penalty'):
            mpp_split.PART_PENALTY = 1.0
            splits = mpp_split.suggest_splits(1100000000, self.channels_with_funds)
            self.assertEqual(2, splits[0].config.number_parts())
        with self.subTest(msg='split payments with intermediate part penalty'):
            mpp_split.PART_PENALTY = 0.3
            splits = mpp_split.suggest_splits(1100000000, self.channels_with_funds)
            self.assertEqual(4, splits[0].config.number_parts())
        with self.subTest(msg='split payments with no part penalty'):
            mpp_split.PART_PENALTY = 0.0
            splits = mpp_split.suggest_splits(1100000000, self.channels_with_funds)
            self.assertEqual(5, splits[0].config.number_parts())

    def test_suggest_splits_single_channel(self):
        if False:
            while True:
                i = 10
        channels_with_funds = {(b'0', b'0'): 1000000000}
        with self.subTest(msg='do a payment with the maximal amount spendable on a single channel'):
            splits = mpp_split.suggest_splits(1000000000, channels_with_funds, exclude_single_part_payments=False)
            self.assertEqual({(b'0', b'0'): [1000000000]}, splits[0].config)
        with self.subTest(msg='test sending an amount greater than what we have available'):
            self.assertRaises(NoPathFound, mpp_split.suggest_splits, *(1100000000, channels_with_funds))
        with self.subTest(msg='test sending a large amount over a single channel in chunks'):
            mpp_split.PART_PENALTY = 0.5
            splits = mpp_split.suggest_splits(1000000000, channels_with_funds, exclude_single_part_payments=False)
            self.assertEqual(2, len(splits[0].config[b'0', b'0']))
        with self.subTest(msg='test sending a large amount over a single channel in chunks'):
            mpp_split.PART_PENALTY = 0.3
            splits = mpp_split.suggest_splits(1000000000, channels_with_funds, exclude_single_part_payments=False)
            self.assertEqual(3, len(splits[0].config[b'0', b'0']))
        with self.subTest(msg='exclude all single channel splits'):
            mpp_split.PART_PENALTY = 0.3
            splits = mpp_split.suggest_splits(1000000000, channels_with_funds, exclude_single_channel_splits=True)
            self.assertEqual(1, len(splits[0].config[b'0', b'0']))