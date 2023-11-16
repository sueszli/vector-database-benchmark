import bisect
import distutils
import math
from copy import copy
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
import logging
from rich.logging import RichHandler
from ciphey.iface import Checker, Config, Cracker, CrackInfo, CrackResult, Decoder, ParamSpec, Searcher, SearchLevel, SearchResult, T, registry
'\nWe are using a tree structure here, because that makes searching and tracing back easier\nAs such, when we encounter another possible parent, we remove that edge\n'

class DuplicateNode(Exception):
    pass

@dataclass
class AuSearchSuccessful(Exception):
    target: 'Node'
    info: str

@dataclass
class Node:
    level: SearchLevel
    parent: Optional['Edge'] = None
    depth: int = 0

    @staticmethod
    def decoding(config: Config, route: Union[Cracker, Decoder], result: Any, source: 'Node') -> 'Node':
        if False:
            return 10
        if not config.cache.mark_ctext(result):
            raise DuplicateNode()
        checker: Checker = config.objs['checker']
        ret = Node(parent=None, level=SearchLevel(name=type(route).__name__.lower(), result=CrackResult(value=result)), depth=source.depth + 1)
        edge = Edge(source=source, route=route, dest=ret)
        ret.parent = edge
        check_res = checker(result)
        if check_res is not None:
            raise AuSearchSuccessful(target=ret, info=check_res)
        return ret

    @staticmethod
    def cracker(config: Config, edge_template: 'Edge', result: CrackResult) -> 'Node':
        if False:
            print('Hello World!')
        if not config.cache.mark_ctext(result.value):
            raise DuplicateNode()
        checker: Checker = config.objs['checker']
        edge = copy(edge_template)
        ret = Node(parent=edge, level=SearchLevel(name=type(edge.route).__name__.lower(), result=result), depth=edge.source.depth + 1)
        edge.dest = ret
        check_res = checker(result.value)
        if check_res is not None:
            raise AuSearchSuccessful(target=ret, info=check_res)
        return ret

    @staticmethod
    def root(config: Config, ctext: Any):
        if False:
            print('Hello World!')
        if not config.cache.mark_ctext(ctext):
            raise DuplicateNode()
        return Node(parent=None, level=SearchLevel.input(ctext))

    def get_path(self):
        if False:
            for i in range(10):
                print('nop')
        if self.parent is None:
            return [self.level]
        return self.parent.source.get_path() + [self.level]

@dataclass
class AusearchEdge:
    success_probability: float
    failure_probability: float
    success_time: float
    failure_time: float

    def __init__(self, success_probability, success_time, failure_time):
        if False:
            for i in range(10):
                print('nop')
        self.success_probability = success_probability
        self.failure_probability = 1.0 - success_probability
        self.success_time = success_time
        self.failure_time = failure_time

@dataclass
class AusearchResult:
    weight: float
    index: int

def calculate_score(info: CrackInfo):
    if False:
        i = 10
        return i + 15
    return info.success_likelihood / (info.success_runtime * info.success_likelihood + info.failure_runtime * (1 - info.success_likelihood))

@dataclass
class Edge:
    source: Node
    route: Union[Cracker, Decoder]
    dest: Optional[Node] = None
    score: Optional[float] = None
PriorityType = TypeVar('PriorityType')

class PriorityWorkQueue(Generic[PriorityType, T]):
    _sorted_priorities: List[PriorityType]
    _queues: Dict[Any, List[T]]

    def add_work(self, priority: PriorityType, work: List[T]) -> None:
        if False:
            for i in range(10):
                print('nop')
        logging.debug(f'Adding work at depth {priority}')
        idx = bisect.bisect_left(self._sorted_priorities, priority)
        if idx == len(self._sorted_priorities) or self._sorted_priorities[idx] != priority:
            self._sorted_priorities.insert(idx, priority)
        self._queues.setdefault(priority, []).extend(work)

    def get_work(self) -> T:
        if False:
            return 10
        best_priority = self._sorted_priorities[0]
        target = self._queues[best_priority]
        ret = target.pop(0)
        if len(target) == 0:
            self._sorted_priorities.pop()
        return ret

    def get_work_chunk(self) -> List[T]:
        if False:
            i = 10
            return i + 15
        'Returns the best work for now'
        if len(self._sorted_priorities) == 0:
            return []
        best_priority = self._sorted_priorities.pop(0)
        return self._queues.pop(best_priority)

    def empty(self):
        if False:
            print('Hello World!')
        return len(self._sorted_priorities) == 0

    def __init__(self):
        if False:
            return 10
        self._sorted_priorities = []
        self._queues = {}

@registry.register
class AuSearch(Searcher):
    work: PriorityWorkQueue[int, Edge]

    @staticmethod
    def get_crackers_for(t: type):
        if False:
            return 10
        return registry[Cracker[t]]

    @lru_cache()
    def get_decoders_for(self, t: type):
        if False:
            print('Hello World!')
        ret = registry[Decoder[t]]
        ret.sort(key=lambda x: x.priority(), reverse=True)
        return ret

    def expand_crackers(self, node: Node) -> None:
        if False:
            while True:
                i = 10
        if node.depth >= self.max_cipher_depth:
            return
        res = node.level.result.value
        additional_work = []
        for i in self.get_crackers_for(type(res)):
            inst = self._config()(i)
            info = inst.getInfo(res)
            if info.success_likelihood < self.p_threshold:
                continue
            additional_work.append(Edge(source=node, route=inst, score=calculate_score(inst.getInfo(res))))
        priority = min(node.depth, self.priority_cap)
        if self.invert_priority:
            priority = -priority
        self.work.add_work(priority, additional_work)

    def expand_decodings(self, node: Node) -> None:
        if False:
            print('Hello World!')
        val = node.level.result.value
        for decoder in self.get_decoders_for(type(val)):
            inst = self._config()(decoder)
            res = inst(val)
            if res is None:
                continue
            try:
                new_node = Node.decoding(config=self._config(), route=inst, result=res, source=node)
            except DuplicateNode:
                continue
            logging.debug('Nesting encodings')
            self.recursive_expand(new_node, False)

    def recursive_expand(self, node: Node, nested: bool=True) -> None:
        if False:
            i = 10
            return i + 15
        if node.depth >= self.max_depth:
            return
        logging.debug(f'Expanding depth {node.depth}')
        self.expand_decodings(node)
        if not nested or self.enable_nested:
            self.expand_crackers(node)

    def search(self, ctext: Any) -> Optional[SearchResult]:
        if False:
            print('Hello World!')
        logging.debug(f"Beginning AuSearch with {('inverted' if self.invert_priority else 'normal')} priority")
        try:
            root = Node.root(self._config(), ctext)
        except DuplicateNode:
            return None
        check_res = self._config().objs['checker'](ctext)
        if check_res is not None:
            return SearchResult(check_res=check_res, path=[root.level])
        try:
            self.recursive_expand(root, False)
            while True:
                if self.work.empty():
                    break
                chunk = self.work.get_work_chunk()
                chunk.sort(key=lambda i: i.score)
                while len(chunk) != 0:
                    logging.debug(f'{len(chunk)} remaining on this level')
                    edge: Edge = chunk.pop(-1)
                    res = edge.route(edge.source.level.result.value)
                    if res is None:
                        continue
                    for i in res:
                        try:
                            node = Node.cracker(config=self._config(), edge_template=edge, result=i)
                            self.recursive_expand(node)
                        except DuplicateNode:
                            continue
        except AuSearchSuccessful as e:
            logging.info('AuSearch succeeded')
            return SearchResult(path=e.target.get_path(), check_res=e.info)
        logging.info('AuSearch failed')

    def __init__(self, config: Config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self._checker: Checker = config.objs['checker']
        self.work = PriorityWorkQueue()
        self.invert_priority = bool(distutils.util.strtobool(self._params()['invert_priority']))
        self.priority_cap = int(self._params()['priority_cap'])
        self.enable_nested = bool(distutils.util.strtobool(self._params()['enable_nested']))
        self.max_cipher_depth = int(self._params()['max_cipher_depth'])
        if self.max_cipher_depth == 0:
            self.max_cipher_depth = math.inf
        self.max_depth = int(self._params()['max_depth'])
        if self.max_depth == 0:
            self.max_depth = math.inf
        self.p_threshold = float(self._params()['p_threshold'])

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            i = 10
            return i + 15
        return {'enable_nested': ParamSpec(req=False, desc='Enables nested ciphers. Incredibly slow, and not guaranteed to terminate', default='False'), 'invert_priority': ParamSpec(req=False, desc='Causes more complex encodings to be looked at first. Good for deeply buried encodings.', default='False'), 'max_cipher_depth': ParamSpec(req=False, desc='The depth at which we stop trying to crack ciphers. Set to 0 to disable', default='0'), 'max_depth': ParamSpec(req=False, desc='The depth at which we give up. Set to 0 to disable', default='0'), 'priority_cap': ParamSpec(req=False, desc='Sets the maximum depth before we give up ordering items.', default='2'), 'p_threshold': ParamSpec(req=False, desc='Will skip any crackers which have less than this likelihood of succeeding. Set to 0 to disable', default='0.01')}