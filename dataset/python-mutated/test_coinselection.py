from types import GeneratorType
from lbry.testcase import AsyncioTestCase
from lbry.wallet import Ledger, Database, Headers
from lbry.wallet.coinselection import CoinSelector, MAXIMUM_TRIES
from lbry.constants import CENT
from tests.unit.wallet.test_transaction import get_output as utxo
NULL_HASH = b'\x00' * 32

def search(*args, **kwargs):
    if False:
        return 10
    selection = CoinSelector(*args[1:], **kwargs).select(args[0], 'branch_and_bound')
    return [o.txo.amount for o in selection] if selection else selection

class BaseSelectionTestCase(AsyncioTestCase):

    async def asyncSetUp(self):
        self.ledger = Ledger({'db': Database(':memory:'), 'headers': Headers(':memory:')})
        await self.ledger.db.open()

    async def asyncTearDown(self):
        await self.ledger.db.close()

    def estimates(self, *args):
        if False:
            i = 10
            return i + 15
        txos = args[0] if isinstance(args[0], (GeneratorType, list)) else args
        return [txo.get_estimator(self.ledger) for txo in txos]

class TestCoinSelectionTests(BaseSelectionTestCase):

    def test_empty_coins(self):
        if False:
            i = 10
            return i + 15
        self.assertListEqual(CoinSelector(0, 0).select([]), [])

    def test_skip_binary_search_if_total_not_enough(self):
        if False:
            return 10
        fee = utxo(CENT).get_estimator(self.ledger).fee
        big_pool = self.estimates((utxo(CENT + fee) for _ in range(100)))
        selector = CoinSelector(101 * CENT, 0)
        self.assertListEqual(selector.select(big_pool), [])
        self.assertEqual(selector.tries, 0)
        selector = CoinSelector(100 * CENT, 0)
        self.assertEqual(len(selector.select(big_pool)), 100)
        self.assertEqual(selector.tries, 201)

    def test_exact_match(self):
        if False:
            print('Hello World!')
        fee = utxo(CENT).get_estimator(self.ledger).fee
        utxo_pool = self.estimates(utxo(CENT + fee), utxo(CENT), utxo(CENT - fee))
        selector = CoinSelector(CENT, 0)
        match = selector.select(utxo_pool)
        self.assertListEqual([CENT + fee], [c.txo.amount for c in match])
        self.assertTrue(selector.exact_match)

    def test_random_draw(self):
        if False:
            for i in range(10):
                print('nop')
        utxo_pool = self.estimates(utxo(2 * CENT), utxo(3 * CENT), utxo(4 * CENT))
        selector = CoinSelector(CENT, 0, '\x00')
        match = selector.select(utxo_pool)
        self.assertListEqual([2 * CENT], [c.txo.amount for c in match])
        self.assertFalse(selector.exact_match)

    def test_pick(self):
        if False:
            for i in range(10):
                print('nop')
        utxo_pool = self.estimates(utxo(1 * CENT), utxo(1 * CENT), utxo(3 * CENT), utxo(5 * CENT), utxo(10 * CENT))
        selector = CoinSelector(3 * CENT, 0)
        match = selector.select(utxo_pool)
        self.assertListEqual([5 * CENT], [c.txo.amount for c in match])

    def test_confirmed_strategies(self):
        if False:
            while True:
                i = 10
        utxo_pool = self.estimates(utxo(11 * CENT, height=5), utxo(11 * CENT, height=0), utxo(11 * CENT, height=-2), utxo(11 * CENT, height=5))
        match = CoinSelector(20 * CENT, 0).select(utxo_pool, 'only_confirmed')
        self.assertListEqual([5, 5], [c.txo.tx_ref.height for c in match])
        match = CoinSelector(25 * CENT, 0).select(utxo_pool, 'only_confirmed')
        self.assertListEqual([], [c.txo.tx_ref.height for c in match])
        match = CoinSelector(20 * CENT, 0).select(utxo_pool, 'prefer_confirmed')
        self.assertListEqual([5, 5], [c.txo.tx_ref.height for c in match])
        match = CoinSelector(25 * CENT, 0, '\x00').select(utxo_pool, 'prefer_confirmed')
        self.assertListEqual([5, 0, -2], [c.txo.tx_ref.height for c in match])

class TestOfficialBitcoinCoinSelectionTests(BaseSelectionTestCase):

    def make_hard_case(self, utxos):
        if False:
            print('Hello World!')
        target = 0
        utxo_pool = []
        for i in range(utxos):
            amount = 1 << utxos + i
            target += amount
            utxo_pool.append(utxo(amount))
            utxo_pool.append(utxo(amount + (1 << utxos - 1 - i)))
        return (self.estimates(utxo_pool), target)

    def test_branch_and_bound_coin_selection(self):
        if False:
            while True:
                i = 10
        self.ledger.fee_per_byte = 0
        utxo_pool = self.estimates(utxo(1 * CENT), utxo(2 * CENT), utxo(3 * CENT), utxo(4 * CENT))
        self.assertListEqual([1 * CENT], search(utxo_pool, 1 * CENT, 0.5 * CENT))
        self.assertListEqual([2 * CENT], search(utxo_pool, 2 * CENT, 0.5 * CENT))
        self.assertListEqual([3 * CENT, 2 * CENT], search(utxo_pool, 5 * CENT, 0.5 * CENT))
        self.assertListEqual([], search(utxo_pool, 11 * CENT, 0.5 * CENT))
        utxo_pool += self.estimates(utxo(5 * CENT))
        self.assertListEqual([4 * CENT, 3 * CENT, 2 * CENT, 1 * CENT], search(utxo_pool, 10 * CENT, 0.5 * CENT))
        self.assertListEqual([4 * CENT, 3 * CENT, 2 * CENT, 1 * CENT], search(utxo_pool, 10 * CENT, 5000))
        self.assertListEqual(search(utxo_pool, 0.25 * CENT, 0.5 * CENT), [])
        (utxo_pool, target) = self.make_hard_case(17)
        selector = CoinSelector(target, 0)
        self.assertListEqual(selector.select(utxo_pool, 'branch_and_bound'), [])
        self.assertEqual(selector.tries, MAXIMUM_TRIES)
        (utxo_pool, target) = self.make_hard_case(14)
        self.assertIsNotNone(search(utxo_pool, target, 0))
        utxo_pool = self.estimates([utxo(7 * CENT), utxo(7 * CENT), utxo(7 * CENT), utxo(7 * CENT), utxo(2 * CENT)] + [utxo(5 * CENT)] * 50000)
        self.assertListEqual([7 * CENT, 7 * CENT, 7 * CENT, 7 * CENT, 2 * CENT], search(utxo_pool, 30 * CENT, 5000))
        utxo_pool = self.estimates((utxo(i * CENT) for i in range(5, 21)))
        for _ in range(100):
            self.assertListEqual(search(utxo_pool, 1 * CENT, 2 * CENT), [])