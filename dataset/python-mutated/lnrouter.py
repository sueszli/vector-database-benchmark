import queue
from collections import defaultdict
from typing import Sequence, Tuple, Optional, Dict, TYPE_CHECKING, Set
import time
import threading
from threading import RLock
import attr
from math import inf
from .util import profiler, with_lock
from .logging import Logger
from .lnutil import NUM_MAX_EDGES_IN_PAYMENT_PATH, ShortChannelID, LnFeatures, NBLOCK_CLTV_DELTA_TOO_FAR_INTO_FUTURE, PaymentFeeBudget
from .channel_db import ChannelDB, Policy, NodeInfo
if TYPE_CHECKING:
    from .lnchannel import Channel
DEFAULT_PENALTY_BASE_MSAT = 500
DEFAULT_PENALTY_PROPORTIONAL_MILLIONTH = 100
HINT_DURATION = 3600

class NoChannelPolicy(Exception):

    def __init__(self, short_channel_id: bytes):
        if False:
            print('Hello World!')
        short_channel_id = ShortChannelID.normalize(short_channel_id)
        super().__init__(f'cannot find channel policy for short_channel_id: {short_channel_id}')

class LNPathInconsistent(Exception):
    pass

def fee_for_edge_msat(forwarded_amount_msat: int, fee_base_msat: int, fee_proportional_millionths: int) -> int:
    if False:
        print('Hello World!')
    return fee_base_msat + forwarded_amount_msat * fee_proportional_millionths // 1000000

@attr.s(slots=True)
class PathEdge:
    start_node = attr.ib(type=bytes, kw_only=True, repr=lambda val: val.hex())
    end_node = attr.ib(type=bytes, kw_only=True, repr=lambda val: val.hex())
    short_channel_id = attr.ib(type=ShortChannelID, kw_only=True, repr=lambda val: str(val))

    @property
    def node_id(self) -> bytes:
        if False:
            while True:
                i = 10
        return self.end_node

@attr.s
class RouteEdge(PathEdge):
    fee_base_msat = attr.ib(type=int, kw_only=True)
    fee_proportional_millionths = attr.ib(type=int, kw_only=True)
    cltv_delta = attr.ib(type=int, kw_only=True)
    node_features = attr.ib(type=int, kw_only=True, repr=lambda val: str(int(val)))

    def fee_for_edge(self, amount_msat: int) -> int:
        if False:
            return 10
        return fee_for_edge_msat(forwarded_amount_msat=amount_msat, fee_base_msat=self.fee_base_msat, fee_proportional_millionths=self.fee_proportional_millionths)

    @classmethod
    def from_channel_policy(cls, *, channel_policy: 'Policy', short_channel_id: bytes, start_node: bytes, end_node: bytes, node_info: Optional[NodeInfo]) -> 'RouteEdge':
        if False:
            while True:
                i = 10
        assert isinstance(short_channel_id, bytes)
        assert type(start_node) is bytes
        assert type(end_node) is bytes
        return RouteEdge(start_node=start_node, end_node=end_node, short_channel_id=ShortChannelID.normalize(short_channel_id), fee_base_msat=channel_policy.fee_base_msat, fee_proportional_millionths=channel_policy.fee_proportional_millionths, cltv_delta=channel_policy.cltv_delta, node_features=node_info.features if node_info else 0)

    def is_sane_to_use(self, amount_msat: int) -> bool:
        if False:
            i = 10
            return i + 15
        if self.cltv_delta > 14 * 144:
            return False
        total_fee = self.fee_for_edge(amount_msat)
        if total_fee > get_default_fee_budget_msat(invoice_amount_msat=amount_msat):
            return False
        return True

    def has_feature_varonion(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        features = LnFeatures(self.node_features)
        return features.supports(LnFeatures.VAR_ONION_OPT)

    def is_trampoline(self) -> bool:
        if False:
            print('Hello World!')
        return False

@attr.s
class TrampolineEdge(RouteEdge):
    invoice_routing_info = attr.ib(type=bytes, default=None)
    invoice_features = attr.ib(type=int, default=None)
    short_channel_id = attr.ib(default=ShortChannelID(8), repr=lambda val: str(val))

    def is_trampoline(self):
        if False:
            for i in range(10):
                print('nop')
        return True
LNPaymentPath = Sequence[PathEdge]
LNPaymentRoute = Sequence[RouteEdge]
LNPaymentTRoute = Sequence[TrampolineEdge]

def is_route_within_budget(route: LNPaymentRoute, *, budget: PaymentFeeBudget, amount_msat_for_dest: int, cltv_delta_for_dest: int) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Run some sanity checks on the whole route, before attempting to use it.\n    called when we are paying; so e.g. lower cltv is better\n    '
    if len(route) > NUM_MAX_EDGES_IN_PAYMENT_PATH:
        return False
    amt = amount_msat_for_dest
    cltv_cost_of_route = 0
    for route_edge in reversed(route[1:]):
        if not route_edge.is_sane_to_use(amt):
            return False
        amt += route_edge.fee_for_edge(amt)
        cltv_cost_of_route += route_edge.cltv_delta
    fee_cost = amt - amount_msat_for_dest
    if cltv_cost_of_route > budget.cltv:
        return False
    if fee_cost > budget.fee_msat:
        return False
    total_cltv_delta = cltv_cost_of_route + cltv_delta_for_dest
    if total_cltv_delta > NBLOCK_CLTV_DELTA_TOO_FAR_INTO_FUTURE:
        return False
    return True

def get_default_fee_budget_msat(*, invoice_amount_msat: int) -> int:
    if False:
        i = 10
        return i + 15
    return max(5000, invoice_amount_msat // 100)

class LiquidityHint:
    """Encodes the amounts that can and cannot be sent over the direction of a
    channel.

    A LiquidityHint is the value of a dict, which is keyed to node ids and the
    channel.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._can_send_forward = None
        self._cannot_send_forward = None
        self._can_send_backward = None
        self._cannot_send_backward = None
        self.hint_timestamp = 0
        self._inflight_htlcs_forward = 0
        self._inflight_htlcs_backward = 0

    def is_hint_invalid(self) -> bool:
        if False:
            return 10
        now = int(time.time())
        return now - self.hint_timestamp > HINT_DURATION

    @property
    def can_send_forward(self):
        if False:
            while True:
                i = 10
        return None if self.is_hint_invalid() else self._can_send_forward

    @can_send_forward.setter
    def can_send_forward(self, amount):
        if False:
            for i in range(10):
                print('nop')
        if self._can_send_forward and self._can_send_forward > amount:
            return
        self._can_send_forward = amount
        if self._cannot_send_forward and self._can_send_forward > self._cannot_send_forward:
            self._cannot_send_forward = None

    @property
    def can_send_backward(self):
        if False:
            for i in range(10):
                print('nop')
        return None if self.is_hint_invalid() else self._can_send_backward

    @can_send_backward.setter
    def can_send_backward(self, amount):
        if False:
            for i in range(10):
                print('nop')
        if self._can_send_backward and self._can_send_backward > amount:
            return
        self._can_send_backward = amount
        if self._cannot_send_backward and self._can_send_backward > self._cannot_send_backward:
            self._cannot_send_backward = None

    @property
    def cannot_send_forward(self):
        if False:
            i = 10
            return i + 15
        return None if self.is_hint_invalid() else self._cannot_send_forward

    @cannot_send_forward.setter
    def cannot_send_forward(self, amount):
        if False:
            while True:
                i = 10
        if self._cannot_send_forward and self._cannot_send_forward < amount:
            return
        self._cannot_send_forward = amount
        if self._can_send_forward and self._can_send_forward > self._cannot_send_forward:
            self._can_send_forward = None
        self.can_send_backward = amount

    @property
    def cannot_send_backward(self):
        if False:
            while True:
                i = 10
        return None if self.is_hint_invalid() else self._cannot_send_backward

    @cannot_send_backward.setter
    def cannot_send_backward(self, amount):
        if False:
            return 10
        if self._cannot_send_backward and self._cannot_send_backward < amount:
            return
        self._cannot_send_backward = amount
        if self._can_send_backward and self._can_send_backward > self._cannot_send_backward:
            self._can_send_backward = None
        self.can_send_forward = amount

    def can_send(self, is_forward_direction: bool):
        if False:
            print('Hello World!')
        if is_forward_direction:
            return self.can_send_forward
        else:
            return self.can_send_backward

    def cannot_send(self, is_forward_direction: bool):
        if False:
            print('Hello World!')
        if is_forward_direction:
            return self.cannot_send_forward
        else:
            return self.cannot_send_backward

    def update_can_send(self, is_forward_direction: bool, amount: int):
        if False:
            while True:
                i = 10
        self.hint_timestamp = int(time.time())
        if is_forward_direction:
            self.can_send_forward = amount
        else:
            self.can_send_backward = amount

    def update_cannot_send(self, is_forward_direction: bool, amount: int):
        if False:
            i = 10
            return i + 15
        self.hint_timestamp = int(time.time())
        if is_forward_direction:
            self.cannot_send_forward = amount
        else:
            self.cannot_send_backward = amount

    def num_inflight_htlcs(self, is_forward_direction: bool) -> int:
        if False:
            for i in range(10):
                print('nop')
        if is_forward_direction:
            return self._inflight_htlcs_forward
        else:
            return self._inflight_htlcs_backward

    def add_htlc(self, is_forward_direction: bool):
        if False:
            while True:
                i = 10
        if is_forward_direction:
            self._inflight_htlcs_forward += 1
        else:
            self._inflight_htlcs_backward += 1

    def remove_htlc(self, is_forward_direction: bool):
        if False:
            i = 10
            return i + 15
        if is_forward_direction:
            self._inflight_htlcs_forward = max(0, self._inflight_htlcs_forward - 1)
        else:
            self._inflight_htlcs_backward = max(0, self._inflight_htlcs_forward - 1)

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'forward: can send: {self._can_send_forward} msat, cannot send: {self._cannot_send_forward} msat, htlcs: {self._inflight_htlcs_forward}\nbackward: can send: {self._can_send_backward} msat, cannot send: {self._cannot_send_backward} msat, htlcs: {self._inflight_htlcs_backward}\n'

class LiquidityHintMgr:
    """Implements liquidity hints for channels in the graph.

    This class can be used to update liquidity information about channels in the
    graph. Implements a penalty function for edge weighting in the pathfinding
    algorithm that favors channels which can route payments and penalizes
    channels that cannot.
    """

    def __init__(self):
        if False:
            return 10
        self.lock = RLock()
        self._liquidity_hints: Dict[ShortChannelID, LiquidityHint] = {}

    @with_lock
    def get_hint(self, channel_id: ShortChannelID) -> LiquidityHint:
        if False:
            for i in range(10):
                print('nop')
        hint = self._liquidity_hints.get(channel_id)
        if not hint:
            hint = LiquidityHint()
            self._liquidity_hints[channel_id] = hint
        return hint

    @with_lock
    def update_can_send(self, node_from: bytes, node_to: bytes, channel_id: ShortChannelID, amount: int):
        if False:
            for i in range(10):
                print('nop')
        hint = self.get_hint(channel_id)
        hint.update_can_send(node_from < node_to, amount)

    @with_lock
    def update_cannot_send(self, node_from: bytes, node_to: bytes, channel_id: ShortChannelID, amount: int):
        if False:
            return 10
        hint = self.get_hint(channel_id)
        hint.update_cannot_send(node_from < node_to, amount)

    @with_lock
    def add_htlc(self, node_from: bytes, node_to: bytes, channel_id: ShortChannelID):
        if False:
            return 10
        hint = self.get_hint(channel_id)
        hint.add_htlc(node_from < node_to)

    @with_lock
    def remove_htlc(self, node_from: bytes, node_to: bytes, channel_id: ShortChannelID):
        if False:
            return 10
        hint = self.get_hint(channel_id)
        hint.remove_htlc(node_from < node_to)

    def penalty(self, node_from: bytes, node_to: bytes, channel_id: ShortChannelID, amount: int) -> float:
        if False:
            print('Hello World!')
        "Gives a penalty when sending from node1 to node2 over channel_id with an\n        amount in units of millisatoshi.\n\n        The penalty depends on the can_send and cannot_send values that was\n        possibly recorded in previous payment attempts.\n\n        A channel that can send an amount is assigned a penalty of zero, a\n        channel that cannot send an amount is assigned an infinite penalty.\n        If the sending amount lies between can_send and cannot_send, there's\n        uncertainty and we give a default penalty. The default penalty\n        serves the function of giving a positive offset (the Dijkstra\n        algorithm doesn't work with negative weights), from which we can discount\n        from. There is a competition between low-fee channels and channels where\n        we know with some certainty that they can support a payment. The penalty\n        ultimately boils down to: how much more fees do we want to pay for\n        certainty of payment success? This can be tuned via DEFAULT_PENALTY_BASE_MSAT\n        and DEFAULT_PENALTY_PROPORTIONAL_MILLIONTH. A base _and_ relative penalty\n        was chosen such that the penalty will be able to compete with the regular\n        base and relative fees.\n        "
        hint = self._liquidity_hints.get(channel_id)
        if not hint:
            (can_send, cannot_send, num_inflight_htlcs) = (None, None, 0)
        else:
            can_send = hint.can_send(node_from < node_to)
            cannot_send = hint.cannot_send(node_from < node_to)
            num_inflight_htlcs = hint.num_inflight_htlcs(node_from < node_to)
        if cannot_send is not None and amount >= cannot_send:
            return inf
        if can_send is not None and amount <= can_send:
            return 0
        success_fee = fee_for_edge_msat(amount, DEFAULT_PENALTY_BASE_MSAT, DEFAULT_PENALTY_PROPORTIONAL_MILLIONTH)
        inflight_htlc_fee = num_inflight_htlcs * success_fee
        return success_fee + inflight_htlc_fee

    @with_lock
    def reset_liquidity_hints(self):
        if False:
            i = 10
            return i + 15
        for (k, v) in self._liquidity_hints.items():
            v.hint_timestamp = 0

    def __repr__(self):
        if False:
            while True:
                i = 10
        string = 'liquidity hints:\n'
        if self._liquidity_hints:
            for (k, v) in self._liquidity_hints.items():
                string += f'{k}: {v}\n'
        return string

class LNPathFinder(Logger):

    def __init__(self, channel_db: ChannelDB):
        if False:
            print('Hello World!')
        Logger.__init__(self)
        self.channel_db = channel_db
        self.liquidity_hints = LiquidityHintMgr()
        self._edge_blacklist = dict()
        self._blacklist_lock = threading.Lock()

    def _is_edge_blacklisted(self, short_channel_id: ShortChannelID, *, now: int) -> bool:
        if False:
            while True:
                i = 10
        blacklist_expiration = self._edge_blacklist.get(short_channel_id)
        if blacklist_expiration is None:
            return False
        if blacklist_expiration < now:
            return False
        return True

    def add_edge_to_blacklist(self, short_channel_id: ShortChannelID, *, now: int=None, duration: int=3600) -> None:
        if False:
            for i in range(10):
                print('nop')
        if now is None:
            now = int(time.time())
        with self._blacklist_lock:
            blacklist_expiration = self._edge_blacklist.get(short_channel_id, 0)
            self._edge_blacklist[short_channel_id] = max(blacklist_expiration, now + duration)

    def clear_blacklist(self):
        if False:
            while True:
                i = 10
        with self._blacklist_lock:
            self._edge_blacklist = dict()

    def update_liquidity_hints(self, route: LNPaymentRoute, amount_msat: int, failing_channel: ShortChannelID=None):
        if False:
            while True:
                i = 10
        for r in route:
            if r.short_channel_id != failing_channel:
                self.logger.info(f'report {r.short_channel_id} to be able to forward {amount_msat} msat')
                self.liquidity_hints.update_can_send(r.start_node, r.end_node, r.short_channel_id, amount_msat)
            else:
                self.logger.info(f'report {r.short_channel_id} to be unable to forward {amount_msat} msat')
                self.liquidity_hints.update_cannot_send(r.start_node, r.end_node, r.short_channel_id, amount_msat)
                break
        else:
            assert failing_channel is None

    def update_inflight_htlcs(self, route: LNPaymentRoute, add_htlcs: bool):
        if False:
            for i in range(10):
                print('nop')
        self.logger.info(f"{('Adding' if add_htlcs else 'Removing')} inflight htlcs to graph (liquidity hints).")
        for r in route:
            if add_htlcs:
                self.liquidity_hints.add_htlc(r.start_node, r.end_node, r.short_channel_id)
            else:
                self.liquidity_hints.remove_htlc(r.start_node, r.end_node, r.short_channel_id)

    def _edge_cost(self, *, short_channel_id: ShortChannelID, start_node: bytes, end_node: bytes, payment_amt_msat: int, ignore_costs=False, is_mine=False, my_channels: Dict[ShortChannelID, 'Channel']=None, private_route_edges: Dict[ShortChannelID, RouteEdge]=None, now: int) -> Tuple[float, int]:
        if False:
            while True:
                i = 10
        'Heuristic cost (distance metric) of going through a channel.\n        Returns (heuristic_cost, fee_for_edge_msat).\n        '
        if self._is_edge_blacklisted(short_channel_id, now=now):
            return (float('inf'), 0)
        if private_route_edges is None:
            private_route_edges = {}
        channel_info = self.channel_db.get_channel_info(short_channel_id, my_channels=my_channels, private_route_edges=private_route_edges)
        if channel_info is None:
            return (float('inf'), 0)
        channel_policy = self.channel_db.get_policy_for_node(short_channel_id, start_node, my_channels=my_channels, private_route_edges=private_route_edges, now=now)
        if channel_policy is None:
            return (float('inf'), 0)
        channel_policy_backwards = self.channel_db.get_policy_for_node(short_channel_id, end_node, my_channels=my_channels, private_route_edges=private_route_edges, now=now)
        if channel_policy_backwards is None and (not is_mine) and (short_channel_id not in private_route_edges):
            return (float('inf'), 0)
        if channel_policy.is_disabled():
            return (float('inf'), 0)
        if payment_amt_msat < channel_policy.htlc_minimum_msat:
            return (float('inf'), 0)
        if channel_info.capacity_sat is not None and payment_amt_msat // 1000 > channel_info.capacity_sat:
            return (float('inf'), 0)
        if channel_policy.htlc_maximum_msat is not None and payment_amt_msat > channel_policy.htlc_maximum_msat:
            return (float('inf'), 0)
        route_edge = private_route_edges.get(short_channel_id, None)
        if route_edge is None:
            node_info = self.channel_db.get_node_info_for_node_id(node_id=end_node)
            if node_info:
                node_features = LnFeatures(node_info.features)
                if not node_features.supports(LnFeatures.VAR_ONION_OPT):
                    return (float('inf'), 0)
            route_edge = RouteEdge.from_channel_policy(channel_policy=channel_policy, short_channel_id=short_channel_id, start_node=start_node, end_node=end_node, node_info=node_info)
        if not route_edge.is_sane_to_use(payment_amt_msat):
            return (float('inf'), 0)
        if ignore_costs:
            return (DEFAULT_PENALTY_BASE_MSAT, 0)
        fee_msat = route_edge.fee_for_edge(payment_amt_msat)
        cltv_cost = route_edge.cltv_delta * payment_amt_msat * 15 / 1000000000
        liquidity_penalty = self.liquidity_hints.penalty(start_node, end_node, short_channel_id, payment_amt_msat)
        overall_cost = fee_msat + cltv_cost + liquidity_penalty
        return (overall_cost, fee_msat)

    def get_shortest_path_hops(self, *, nodeA: bytes, nodeB: bytes, invoice_amount_msat: int, my_sending_channels: Dict[ShortChannelID, 'Channel']=None, private_route_edges: Dict[ShortChannelID, RouteEdge]=None) -> Dict[bytes, PathEdge]:
        if False:
            i = 10
            return i + 15
        distance_from_start = defaultdict(lambda : float('inf'))
        distance_from_start[nodeB] = 0
        previous_hops = {}
        nodes_to_explore = queue.PriorityQueue()
        nodes_to_explore.put((0, invoice_amount_msat, nodeB))
        now = int(time.time())
        while nodes_to_explore.qsize() > 0:
            (dist_to_edge_endnode, amount_msat, edge_endnode) = nodes_to_explore.get()
            if edge_endnode == nodeA and previous_hops:
                self.logger.info('found a path')
                break
            if dist_to_edge_endnode != distance_from_start[edge_endnode]:
                continue
            if nodeA == nodeB:
                if not previous_hops:
                    channels_for_endnode = self.channel_db.get_channels_for_node(edge_endnode, my_channels={}, private_route_edges=private_route_edges)
                else:
                    channels_for_endnode = self.channel_db.get_channels_for_node(edge_endnode, my_channels=my_sending_channels, private_route_edges={})
            else:
                channels_for_endnode = self.channel_db.get_channels_for_node(edge_endnode, my_channels=my_sending_channels, private_route_edges=private_route_edges)
            for edge_channel_id in channels_for_endnode:
                assert isinstance(edge_channel_id, bytes)
                if self._is_edge_blacklisted(edge_channel_id, now=now):
                    continue
                channel_info = self.channel_db.get_channel_info(edge_channel_id, my_channels=my_sending_channels, private_route_edges=private_route_edges)
                if channel_info is None:
                    continue
                edge_startnode = channel_info.node2_id if channel_info.node1_id == edge_endnode else channel_info.node1_id
                is_mine = edge_channel_id in my_sending_channels
                if is_mine:
                    if edge_startnode == nodeA:
                        if not my_sending_channels[edge_channel_id].can_pay(amount_msat, check_frozen=True):
                            continue
                (edge_cost, fee_for_edge_msat) = self._edge_cost(short_channel_id=edge_channel_id, start_node=edge_startnode, end_node=edge_endnode, payment_amt_msat=amount_msat, ignore_costs=edge_startnode == nodeA, is_mine=is_mine, my_channels=my_sending_channels, private_route_edges=private_route_edges, now=now)
                alt_dist_to_neighbour = distance_from_start[edge_endnode] + edge_cost
                if alt_dist_to_neighbour < distance_from_start[edge_startnode]:
                    distance_from_start[edge_startnode] = alt_dist_to_neighbour
                    previous_hops[edge_startnode] = PathEdge(start_node=edge_startnode, end_node=edge_endnode, short_channel_id=ShortChannelID(edge_channel_id))
                    amount_to_forward_msat = amount_msat + fee_for_edge_msat
                    nodes_to_explore.put((alt_dist_to_neighbour, amount_to_forward_msat, edge_startnode))
            if edge_endnode == nodeB and nodeA == nodeB:
                distance_from_start[edge_endnode] = float('inf')
        return previous_hops

    @profiler
    def find_path_for_payment(self, *, nodeA: bytes, nodeB: bytes, invoice_amount_msat: int, my_sending_channels: Dict[ShortChannelID, 'Channel']=None, private_route_edges: Dict[ShortChannelID, RouteEdge]=None) -> Optional[LNPaymentPath]:
        if False:
            while True:
                i = 10
        'Return a path from nodeA to nodeB.'
        assert type(nodeA) is bytes
        assert type(nodeB) is bytes
        assert type(invoice_amount_msat) is int
        if my_sending_channels is None:
            my_sending_channels = {}
        previous_hops = self.get_shortest_path_hops(nodeA=nodeA, nodeB=nodeB, invoice_amount_msat=invoice_amount_msat, my_sending_channels=my_sending_channels, private_route_edges=private_route_edges)
        if nodeA not in previous_hops:
            return None
        edge_startnode = nodeA
        path = []
        while edge_startnode != nodeB or not path:
            edge = previous_hops[edge_startnode]
            path += [edge]
            edge_startnode = edge.node_id
        return path

    def create_route_from_path(self, path: Optional[LNPaymentPath], *, my_channels: Dict[ShortChannelID, 'Channel']=None, private_route_edges: Dict[ShortChannelID, RouteEdge]=None) -> LNPaymentRoute:
        if False:
            while True:
                i = 10
        if path is None:
            raise Exception('cannot create route from None path')
        if private_route_edges is None:
            private_route_edges = {}
        route = []
        prev_end_node = path[0].start_node
        for path_edge in path:
            short_channel_id = path_edge.short_channel_id
            _endnodes = self.channel_db.get_endnodes_for_chan(short_channel_id, my_channels=my_channels)
            if _endnodes and sorted(_endnodes) != sorted([path_edge.start_node, path_edge.end_node]):
                raise LNPathInconsistent('endpoints of edge inconsistent with short_channel_id')
            if path_edge.start_node != prev_end_node:
                raise LNPathInconsistent('edges do not chain together')
            route_edge = private_route_edges.get(short_channel_id, None)
            if route_edge is None:
                channel_policy = self.channel_db.get_policy_for_node(short_channel_id=short_channel_id, node_id=path_edge.start_node, my_channels=my_channels)
                if channel_policy is None:
                    raise NoChannelPolicy(short_channel_id)
                node_info = self.channel_db.get_node_info_for_node_id(node_id=path_edge.end_node)
                route_edge = RouteEdge.from_channel_policy(channel_policy=channel_policy, short_channel_id=short_channel_id, start_node=path_edge.start_node, end_node=path_edge.end_node, node_info=node_info)
            route.append(route_edge)
            prev_end_node = path_edge.end_node
        return route

    def find_route(self, *, nodeA: bytes, nodeB: bytes, invoice_amount_msat: int, path=None, my_sending_channels: Dict[ShortChannelID, 'Channel']=None, private_route_edges: Dict[ShortChannelID, RouteEdge]=None) -> Optional[LNPaymentRoute]:
        if False:
            i = 10
            return i + 15
        route = None
        if not path:
            path = self.find_path_for_payment(nodeA=nodeA, nodeB=nodeB, invoice_amount_msat=invoice_amount_msat, my_sending_channels=my_sending_channels, private_route_edges=private_route_edges)
        if path:
            route = self.create_route_from_path(path, my_channels=my_sending_channels, private_route_edges=private_route_edges)
        return route