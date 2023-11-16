from collections import defaultdict
from math import floor, log10
from typing import NamedTuple, List, Callable, Sequence, Union, Dict, Tuple, Mapping, Type, TYPE_CHECKING
from decimal import Decimal
from .bitcoin import sha256, COIN, is_address
from .transaction import Transaction, TxOutput, PartialTransaction, PartialTxInput, PartialTxOutput
from .util import NotEnoughFunds
from .logging import Logger
if TYPE_CHECKING:
    from .simple_config import SimpleConfig

class PRNG:

    def __init__(self, seed):
        if False:
            print('Hello World!')
        self.sha = sha256(seed)
        self.pool = bytearray()

    def get_bytes(self, n: int) -> bytes:
        if False:
            i = 10
            return i + 15
        while len(self.pool) < n:
            self.pool.extend(self.sha)
            self.sha = sha256(self.sha)
        (result, self.pool) = (self.pool[:n], self.pool[n:])
        return bytes(result)

    def randint(self, start, end):
        if False:
            print('Hello World!')
        n = end - start
        r = 0
        p = 1
        while p < n:
            r = self.get_bytes(1)[0] + (r << 8)
            p = p << 8
        return start + r % n

    def choice(self, seq):
        if False:
            i = 10
            return i + 15
        return seq[self.randint(0, len(seq))]

    def shuffle(self, x):
        if False:
            print('Hello World!')
        for i in reversed(range(1, len(x))):
            j = self.randint(0, i + 1)
            (x[i], x[j]) = (x[j], x[i])

class Bucket(NamedTuple):
    desc: str
    weight: int
    value: int
    effective_value: int
    coins: List[PartialTxInput]
    min_height: int
    witness: bool

class ScoredCandidate(NamedTuple):
    penalty: float
    tx: PartialTransaction
    buckets: List[Bucket]

def strip_unneeded(bkts: List[Bucket], sufficient_funds) -> List[Bucket]:
    if False:
        i = 10
        return i + 15
    'Remove buckets that are unnecessary in achieving the spend amount'
    if sufficient_funds([], bucket_value_sum=0):
        return []
    bkts = sorted(bkts, key=lambda bkt: bkt.value, reverse=True)
    bucket_value_sum = 0
    for i in range(len(bkts)):
        bucket_value_sum += bkts[i].value
        if sufficient_funds(bkts[:i + 1], bucket_value_sum=bucket_value_sum):
            return bkts[:i + 1]
    raise Exception('keeping all buckets is still not enough')

class CoinChooserBase(Logger):

    def __init__(self, *, enable_output_value_rounding: bool):
        if False:
            print('Hello World!')
        Logger.__init__(self)
        self.enable_output_value_rounding = enable_output_value_rounding

    def keys(self, coins: Sequence[PartialTxInput]) -> Sequence[str]:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def bucketize_coins(self, coins: Sequence[PartialTxInput], *, fee_estimator_vb):
        if False:
            for i in range(10):
                print('nop')
        keys = self.keys(coins)
        buckets = defaultdict(list)
        for (key, coin) in zip(keys, coins):
            buckets[key].append(coin)
        constant_fee = fee_estimator_vb(2000) == fee_estimator_vb(200)

        def make_Bucket(desc: str, coins: List[PartialTxInput]):
            if False:
                print('Hello World!')
            witness = any((coin.is_segwit(guess_for_address=True) for coin in coins))
            weight = sum((Transaction.estimated_input_weight(coin, witness) for coin in coins))
            value = sum((coin.value_sats() for coin in coins))
            min_height = min((coin.block_height for coin in coins))
            assert min_height is not None
            if constant_fee:
                effective_value = value
            else:
                fee = fee_estimator_vb(Decimal(weight) / 4)
                effective_value = value - fee
            return Bucket(desc=desc, weight=weight, value=value, effective_value=effective_value, coins=coins, min_height=min_height, witness=witness)
        return list(map(make_Bucket, buckets.keys(), buckets.values()))

    def penalty_func(self, base_tx, *, tx_from_buckets: Callable[[List[Bucket]], Tuple[PartialTransaction, List[PartialTxOutput]]]) -> Callable[[List[Bucket]], ScoredCandidate]:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def _change_amounts(self, tx: PartialTransaction, count: int, fee_estimator_numchange) -> List[int]:
        if False:
            i = 10
            return i + 15
        output_amounts = [o.value for o in tx.outputs()]
        max_change = max(max(output_amounts) * 1.25, 0.02 * COIN)
        for n in range(1, count + 1):
            change_amount = max(0, tx.get_fee() - fee_estimator_numchange(n))
            if change_amount // n <= max_change:
                break

        def trailing_zeroes(val):
            if False:
                for i in range(10):
                    print('nop')
            s = str(val)
            return len(s) - len(s.rstrip('0'))
        zeroes = [trailing_zeroes(i) for i in output_amounts]
        min_zeroes = min(zeroes)
        max_zeroes = max(zeroes)
        if n > 1:
            zeroes = range(max(0, min_zeroes - 1), max_zeroes + 1 + 1)
        else:
            zeroes = [min_zeroes]
        remaining = change_amount
        amounts = []
        while n > 1:
            average = remaining / n
            amount = self.p.randint(int(average * 0.7), int(average * 1.3))
            precision = min(self.p.choice(zeroes), int(floor(log10(amount))))
            amount = int(round(amount, -precision))
            amounts.append(amount)
            remaining -= amount
            n -= 1
        max_dp_to_round_for_privacy = 2 if self.enable_output_value_rounding else 0
        N = int(pow(10, min(max_dp_to_round_for_privacy, zeroes[0])))
        amount = remaining // N * N
        amounts.append(amount)
        assert sum(amounts) <= change_amount
        return amounts

    def _change_outputs(self, tx: PartialTransaction, change_addrs, fee_estimator_numchange, dust_threshold) -> List[PartialTxOutput]:
        if False:
            return 10
        amounts = self._change_amounts(tx, len(change_addrs), fee_estimator_numchange)
        assert min(amounts) >= 0
        assert len(change_addrs) >= len(amounts)
        assert all([isinstance(amt, int) for amt in amounts])
        amounts = [amount for amount in amounts if amount >= dust_threshold]
        change = [PartialTxOutput.from_address_and_value(addr, amount) for (addr, amount) in zip(change_addrs, amounts)]
        for c in change:
            c.is_change = True
        return change

    def _construct_tx_from_selected_buckets(self, *, buckets: Sequence[Bucket], base_tx: PartialTransaction, change_addrs, fee_estimator_w, dust_threshold, base_weight) -> Tuple[PartialTransaction, List[PartialTxOutput]]:
        if False:
            for i in range(10):
                print('nop')
        tx = PartialTransaction.from_io(base_tx.inputs()[:], base_tx.outputs()[:])
        tx.add_inputs([coin for b in buckets for coin in b.coins])
        tx_weight = self._get_tx_weight(buckets, base_weight=base_weight)
        if not change_addrs:
            change_addrs = [tx.inputs()[0].address]
            assert is_address(change_addrs[0])
        output_weight = 4 * Transaction.estimated_output_size_for_address(change_addrs[0])
        fee_estimator_numchange = lambda count: fee_estimator_w(tx_weight + count * output_weight)
        change = self._change_outputs(tx, change_addrs, fee_estimator_numchange, dust_threshold)
        tx.add_outputs(change)
        return (tx, change)

    def _get_tx_weight(self, buckets: Sequence[Bucket], *, base_weight: int) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Given a collection of buckets, return the total weight of the\n        resulting transaction.\n        base_weight is the weight of the tx that includes the fixed (non-change)\n        outputs and potentially some fixed inputs. Note that the change outputs\n        at this point are not yet known so they are NOT accounted for.\n        '
        total_weight = base_weight + sum((bucket.weight for bucket in buckets))
        is_segwit_tx = any((bucket.witness for bucket in buckets))
        if is_segwit_tx:
            total_weight += 2
            num_legacy_inputs = sum(((not bucket.witness) * len(bucket.coins) for bucket in buckets))
            total_weight += num_legacy_inputs
        return total_weight

    def make_tx(self, *, coins: Sequence[PartialTxInput], inputs: List[PartialTxInput], outputs: List[PartialTxOutput], change_addrs: Sequence[str], fee_estimator_vb: Callable, dust_threshold: int) -> PartialTransaction:
        if False:
            for i in range(10):
                print('nop')
        'Select unspent coins to spend to pay outputs.  If the change is\n        greater than dust_threshold (after adding the change output to\n        the transaction) it is kept, otherwise none is sent and it is\n        added to the transaction fee.\n\n        `inputs` and `outputs` are guaranteed to be a subset of the\n        inputs and outputs of the resulting transaction.\n        `coins` are further UTXOs we can choose from.\n\n        Note: fee_estimator_vb expects virtual bytes\n        '
        assert outputs, 'tx outputs cannot be empty'
        utxos = [c.prevout.serialize_to_network() for c in coins]
        self.p = PRNG(b''.join(sorted(utxos)))
        base_tx = PartialTransaction.from_io(inputs[:], outputs[:])
        input_value = base_tx.input_value()
        base_weight = base_tx.estimated_weight()
        spent_amount = base_tx.output_value()

        def fee_estimator_w(weight):
            if False:
                while True:
                    i = 10
            return fee_estimator_vb(Transaction.virtual_size_from_weight(weight))

        def sufficient_funds(buckets, *, bucket_value_sum):
            if False:
                return 10
            'Given a list of buckets, return True if it has enough\n            value to pay for the transaction'
            total_input = input_value + bucket_value_sum
            if total_input < spent_amount:
                return False
            if not buckets and (not inputs):
                return False
            total_weight = self._get_tx_weight(buckets, base_weight=base_weight)
            return total_input >= spent_amount + fee_estimator_w(total_weight)

        def tx_from_buckets(buckets):
            if False:
                i = 10
                return i + 15
            return self._construct_tx_from_selected_buckets(buckets=buckets, base_tx=base_tx, change_addrs=change_addrs, fee_estimator_w=fee_estimator_w, dust_threshold=dust_threshold, base_weight=base_weight)
        all_buckets = self.bucketize_coins(coins, fee_estimator_vb=fee_estimator_vb)
        all_buckets = list(filter(lambda b: b.effective_value > 0, all_buckets))
        scored_candidate = self.choose_buckets(all_buckets, sufficient_funds, self.penalty_func(base_tx, tx_from_buckets=tx_from_buckets))
        tx = scored_candidate.tx
        self.logger.info(f'using {len(tx.inputs())} inputs')
        self.logger.info(f'using buckets: {[bucket.desc for bucket in scored_candidate.buckets]}')
        return tx

    def choose_buckets(self, buckets: List[Bucket], sufficient_funds: Callable, penalty_func: Callable[[List[Bucket]], ScoredCandidate]) -> ScoredCandidate:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplemented('To be subclassed')

class CoinChooserRandom(CoinChooserBase):

    def bucket_candidates_any(self, buckets: List[Bucket], sufficient_funds) -> List[List[Bucket]]:
        if False:
            for i in range(10):
                print('nop')
        'Returns a list of bucket sets.'
        if not buckets:
            if sufficient_funds([], bucket_value_sum=0):
                return [[]]
            else:
                raise NotEnoughFunds()
        candidates = set()
        for (n, bucket) in enumerate(buckets):
            if sufficient_funds([bucket], bucket_value_sum=bucket.value):
                candidates.add((n,))
        attempts = min(100, (len(buckets) - 1) * 10 + 1)
        permutation = list(range(len(buckets)))
        for i in range(attempts):
            self.p.shuffle(permutation)
            bkts = []
            bucket_value_sum = 0
            for (count, index) in enumerate(permutation):
                bucket = buckets[index]
                bkts.append(bucket)
                bucket_value_sum += bucket.value
                if sufficient_funds(bkts, bucket_value_sum=bucket_value_sum):
                    candidates.add(tuple(sorted(permutation[:count + 1])))
                    break
            else:
                raise NotEnoughFunds()
        candidates = [[buckets[n] for n in c] for c in candidates]
        return [strip_unneeded(c, sufficient_funds) for c in candidates]

    def bucket_candidates_prefer_confirmed(self, buckets: List[Bucket], sufficient_funds) -> List[List[Bucket]]:
        if False:
            i = 10
            return i + 15
        'Returns a list of bucket sets preferring confirmed coins.\n\n        Any bucket can be:\n        1. "confirmed" if it only contains confirmed coins; else\n        2. "unconfirmed" if it does not contain coins with unconfirmed parents\n        3. other: e.g. "unconfirmed parent" or "local"\n\n        This method tries to only use buckets of type 1, and if the coins there\n        are not enough, tries to use the next type but while also selecting\n        all buckets of all previous types.\n        '
        conf_buckets = [bkt for bkt in buckets if bkt.min_height > 0]
        unconf_buckets = [bkt for bkt in buckets if bkt.min_height == 0]
        other_buckets = [bkt for bkt in buckets if bkt.min_height < 0]
        bucket_sets = [conf_buckets, unconf_buckets, other_buckets]
        already_selected_buckets = []
        already_selected_buckets_value_sum = 0
        for bkts_choose_from in bucket_sets:
            try:

                def sfunds(bkts, *, bucket_value_sum, already_selected_buckets_value_sum=already_selected_buckets_value_sum, already_selected_buckets=already_selected_buckets):
                    if False:
                        return 10
                    bucket_value_sum += already_selected_buckets_value_sum
                    return sufficient_funds(already_selected_buckets + bkts, bucket_value_sum=bucket_value_sum)
                candidates = self.bucket_candidates_any(bkts_choose_from, sfunds)
                break
            except NotEnoughFunds:
                already_selected_buckets += bkts_choose_from
                already_selected_buckets_value_sum += sum((bucket.value for bucket in bkts_choose_from))
        else:
            raise NotEnoughFunds()
        candidates = [already_selected_buckets + c for c in candidates]
        return [strip_unneeded(c, sufficient_funds) for c in candidates]

    def choose_buckets(self, buckets, sufficient_funds, penalty_func):
        if False:
            while True:
                i = 10
        candidates = self.bucket_candidates_prefer_confirmed(buckets, sufficient_funds)
        scored_candidates = [penalty_func(cand) for cand in candidates]
        winner = min(scored_candidates, key=lambda x: x.penalty)
        self.logger.info(f'Total number of buckets: {len(buckets)}')
        self.logger.info(f'Num candidates considered: {len(candidates)}. Winning penalty: {winner.penalty}')
        return winner

class CoinChooserPrivacy(CoinChooserRandom):
    """Attempts to better preserve user privacy.
    First, if any coin is spent from a user address, all coins are.
    Compared to spending from other addresses to make up an amount, this reduces
    information leakage about sender holdings.  It also helps to
    reduce blockchain UTXO bloat, and reduce future privacy loss that
    would come from reusing that address' remaining UTXOs.
    Second, it penalizes change that is quite different to the sent amount.
    Third, it penalizes change that is too big.
    """

    def keys(self, coins):
        if False:
            print('Hello World!')
        return [coin.scriptpubkey.hex() for coin in coins]

    def penalty_func(self, base_tx, *, tx_from_buckets):
        if False:
            while True:
                i = 10
        min_change = min((o.value for o in base_tx.outputs())) * 0.75
        max_change = max((o.value for o in base_tx.outputs())) * 1.33

        def penalty(buckets: List[Bucket]) -> ScoredCandidate:
            if False:
                i = 10
                return i + 15
            badness = len(buckets) - 1
            (tx, change_outputs) = tx_from_buckets(buckets)
            change = sum((o.value for o in change_outputs))
            if change == 0:
                pass
            elif change < min_change:
                badness += (min_change - change) / (min_change + 10000)
                if change < COIN / 1000:
                    badness += 1
            elif change > max_change:
                badness += (change - max_change) / (max_change + 10000)
                badness += change / (COIN * 5)
            return ScoredCandidate(badness, tx, buckets)
        return penalty
COIN_CHOOSERS = {'Privacy': CoinChooserPrivacy}

def get_name(config: 'SimpleConfig') -> str:
    if False:
        while True:
            i = 10
    kind = config.WALLET_COIN_CHOOSER_POLICY
    if kind not in COIN_CHOOSERS:
        kind = config.cv.WALLET_COIN_CHOOSER_POLICY.get_default_value()
    return kind

def get_coin_chooser(config: 'SimpleConfig') -> CoinChooserBase:
    if False:
        while True:
            i = 10
    klass = COIN_CHOOSERS[get_name(config)]
    coinchooser = klass(enable_output_value_rounding=config.WALLET_COIN_CHOOSER_OUTPUT_ROUNDING)
    return coinchooser