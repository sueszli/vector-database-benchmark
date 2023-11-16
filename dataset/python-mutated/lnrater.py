"""
lnrater.py contains Lightning Network node rating functionality.
"""
import asyncio
from collections import defaultdict
from pprint import pformat
from random import choices
from statistics import mean, median, stdev
from typing import TYPE_CHECKING, Dict, NamedTuple, Tuple, List, Optional
import sys
import time
from .logging import Logger
from .util import profiler, get_running_loop
from .lnrouter import fee_for_edge_msat
from .lnutil import LnFeatures, ln_compare_features, IncompatibleLightningFeatures
if TYPE_CHECKING:
    from .network import Network
    from .channel_db import Policy, NodeInfo
    from .lnchannel import ShortChannelID
    from .lnworker import LNWallet
MONTH_IN_BLOCKS = 6 * 24 * 30
RATER_UPDATE_TIME_SEC = 10 * 60
FEE_AMOUNT_MSAT = 100000000
EXCLUDE_NUM_CHANNELS = 15
EXCLUDE_MEAN_CAPACITY_MSAT = 1000000000
EXCLUDE_NODE_AGE = 2 * MONTH_IN_BLOCKS
EXCLUDE_MEAN_CHANNEL_AGE = EXCLUDE_NODE_AGE
EXCLUDE_EFFECTIVE_FEE_RATE = 0.0015
EXCLUDE_BLOCKS_LAST_CHANNEL = 3 * MONTH_IN_BLOCKS

class NodeStats(NamedTuple):
    number_channels: int
    total_capacity_msat: int
    median_capacity_msat: float
    mean_capacity_msat: float
    node_age_block_height: int
    mean_channel_age_block_height: float
    blocks_since_last_channel: int
    mean_fee_rate: float

def weighted_sum(numbers: List[float], weights: List[float]) -> float:
    if False:
        i = 10
        return i + 15
    running_sum = 0.0
    for (n, w) in zip(numbers, weights):
        running_sum += n * w
    return running_sum / sum(weights)

class LNRater(Logger):

    def __init__(self, lnworker: 'LNWallet', network: 'Network'):
        if False:
            print('Hello World!')
        'LNRater can be used to suggest nodes to open up channels with.\n\n        The graph is analyzed and some heuristics are applied to sort out nodes\n        that are deemed to be bad routers or unmaintained.\n        '
        Logger.__init__(self)
        self.lnworker = lnworker
        self.network = network
        self._node_stats: Dict[bytes, NodeStats] = {}
        self._node_ratings: Dict[bytes, float] = {}
        self._policies_by_nodes: Dict[bytes, List[Tuple[ShortChannelID, Policy]]] = defaultdict(list)
        self._last_analyzed = 0
        self._last_progress_percent = 0

    def maybe_analyze_graph(self):
        if False:
            print('Hello World!')
        loop = self.network.asyncio_loop
        fut = asyncio.run_coroutine_threadsafe(self._maybe_analyze_graph(), loop)
        fut.result()

    def analyze_graph(self):
        if False:
            i = 10
            return i + 15
        'Forces a graph analysis, e.g., due to external triggers like\n        the graph info reaching 50%.'
        loop = self.network.asyncio_loop
        fut = asyncio.run_coroutine_threadsafe(self._analyze_graph(), loop)
        fut.result()

    async def _maybe_analyze_graph(self):
        """Analyzes the graph when in early sync stage (>30%) or when caching
        time expires."""
        (current_channels, total, progress_percent) = self.network.lngossip.get_sync_progress_estimate()
        if progress_percent is not None or self.network.channel_db.num_nodes > 500:
            progress_percent = progress_percent or 0
            now = time.time()
            if 30 <= progress_percent and progress_percent - self._last_progress_percent >= 10 or self._last_analyzed + RATER_UPDATE_TIME_SEC < now:
                await self._analyze_graph()
                self._last_progress_percent = progress_percent
                self._last_analyzed = now

    async def _analyze_graph(self):
        await self.network.channel_db.data_loaded.wait()
        self._collect_policies_by_node()
        loop = get_running_loop()
        await loop.run_in_executor(None, self._collect_purged_stats)
        self._rate_nodes()
        now = time.time()
        self._last_analyzed = now

    def _collect_policies_by_node(self):
        if False:
            i = 10
            return i + 15
        policies = self.network.channel_db.get_node_policies()
        for (pv, p) in policies.items():
            self._policies_by_nodes[pv[0]].append((pv[1], p))

    @profiler
    def _collect_purged_stats(self):
        if False:
            return 10
        'Traverses through the graph and sorts out nodes.'
        current_height = self.network.get_local_height()
        node_infos = self.network.channel_db.get_node_infos()
        for (n, channel_policies) in self._policies_by_nodes.items():
            try:
                num_channels = len(channel_policies)
                if num_channels < EXCLUDE_NUM_CHANNELS:
                    continue
                block_heights = [p[0].block_height for p in channel_policies]
                node_age_bh = current_height - min(block_heights)
                if node_age_bh < EXCLUDE_NODE_AGE:
                    continue
                mean_channel_age_bh = current_height - mean(block_heights)
                if mean_channel_age_bh < EXCLUDE_MEAN_CHANNEL_AGE:
                    continue
                blocks_since_last_channel = current_height - max(block_heights)
                if blocks_since_last_channel > EXCLUDE_BLOCKS_LAST_CHANNEL:
                    continue
                capacities = [p[1].htlc_maximum_msat for p in channel_policies]
                if None in capacities:
                    continue
                total_capacity = sum(capacities)
                mean_capacity = total_capacity / num_channels if num_channels else 0
                if mean_capacity < EXCLUDE_MEAN_CAPACITY_MSAT:
                    continue
                median_capacity = median(capacities)
                effective_fee_rates = [fee_for_edge_msat(FEE_AMOUNT_MSAT, p[1].fee_base_msat, p[1].fee_proportional_millionths) / FEE_AMOUNT_MSAT for p in channel_policies]
                mean_fees_rate = mean(effective_fee_rates)
                if mean_fees_rate > EXCLUDE_EFFECTIVE_FEE_RATE:
                    continue
                self._node_stats[n] = NodeStats(number_channels=num_channels, total_capacity_msat=total_capacity, median_capacity_msat=median_capacity, mean_capacity_msat=mean_capacity, node_age_block_height=node_age_bh, mean_channel_age_block_height=mean_channel_age_bh, blocks_since_last_channel=blocks_since_last_channel, mean_fee_rate=mean_fees_rate)
            except Exception as e:
                self.logger.exception('Could not use channel policies for calculating statistics.')
                self.logger.debug(pformat(channel_policies))
                continue
        self.logger.info(f'node statistics done, calculated statisticsfor {len(self._node_stats)} nodes')

    def _rate_nodes(self):
        if False:
            return 10
        'Rate nodes by collected statistics.'
        max_capacity = 0
        max_num_chan = 0
        min_fee_rate = float('inf')
        for stats in self._node_stats.values():
            max_capacity = max(max_capacity, stats.total_capacity_msat)
            max_num_chan = max(max_num_chan, stats.number_channels)
            min_fee_rate = min(min_fee_rate, stats.mean_fee_rate)
        for (n, stats) in self._node_stats.items():
            heuristics = []
            heuristics_weights = []
            heuristics.append(stats.number_channels / max_num_chan)
            heuristics_weights.append(0.2)
            heuristics.append(stats.total_capacity_msat / max_capacity)
            heuristics_weights.append(0.8)
            fees = min(1e-06, min_fee_rate) / max(1e-10, stats.mean_fee_rate)
            heuristics.append(fees)
            heuristics_weights.append(1.0)
            self._node_ratings[n] = weighted_sum(heuristics, heuristics_weights)

    def suggest_node_channel_open(self) -> Tuple[bytes, NodeStats]:
        if False:
            while True:
                i = 10
        node_keys = list(self._node_stats.keys())
        node_ratings = list(self._node_ratings.values())
        channel_peers = self.lnworker.channel_peers()
        node_info: Optional['NodeInfo'] = None
        while True:
            pk = choices(node_keys, weights=node_ratings, k=1)[0]
            node_info = self.network.channel_db.get_node_infos().get(pk, None)
            peer_features = LnFeatures(node_info.features)
            try:
                ln_compare_features(self.lnworker.features, peer_features)
            except IncompatibleLightningFeatures as e:
                self.logger.info('suggested node is incompatible')
                continue
            if pk in channel_peers:
                continue
            if self.lnworker.has_conflicting_backup_with(pk):
                continue
            break
        alias = node_info.alias if node_info else 'unknown node alias'
        self.logger.info(f'node rating for {alias}:\n{pformat(self._node_stats[pk])} (score {self._node_ratings[pk]})')
        return (pk, self._node_stats[pk])

    def suggest_peer(self) -> Optional[bytes]:
        if False:
            while True:
                i = 10
        'Suggests a LN node to open a channel with.\n        Returns a node ID (pubkey).\n        '
        self.maybe_analyze_graph()
        if self._node_ratings:
            return self.suggest_node_channel_open()[0]
        else:
            return None