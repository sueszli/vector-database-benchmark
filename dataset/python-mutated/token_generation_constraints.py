"""Implements tracking of constraints for a beam item.

A list of constraints is given as a list of one or more token
sequences, each of length at least one token. For example, for an input sentence

> Die maschinelle Übersetzung ist schwer zu kontrollieren.

We could have the constraints:
* to influence
* hard

There are two implementations:
* OrderedConstraintState: Tracks progress through an ordered list of multitoken constraints.
* UnorderedConstraintState: Tracks progress through an unordered list of multitoken constraints.

The difference is that in the first, the constraints are assumed to be
in order; the algorithm will permit zero or more tokens between them.
In the second, the constraints are not ordered, so many orderings will
be explored.

The same sequence can be present any number of times, and will appear
that many times in the output.
"""
from collections import Counter
from typing import List, Optional, Set, Tuple
import torch

class ConstraintState:

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

def pack_constraints(batch_constraints: List[List[torch.Tensor]]) -> torch.Tensor:
    if False:
        print('Hello World!')
    'Takes a list of list of constraints in tensor form (a list of\n    tensor constraints for each sentence) and transforms it into a\n    packed Tensor. For example, here is a batch of size 3 with 3, 0,\n    and 1 constraints:\n\n        [ [ [3 1 2], [3], [4 5 6 7], ]\n          [],\n          [ [1 8 9 10 1 4 11 12], ]\n        ]\n\n    Its corresponding packed structure is:\n\n        [ [ 3  3  1  2  0  3  0  4  5  6  7  0],\n          [ 0  0  0  0  0  0  0  0  0  0  0  0],\n          [ 1  1  8  9 10  1  4 11 12  0  0  0] ]\n\n    The packed tensor has shape (batch size, maxlen), where\n    maxlen is defined below. Each row contains concatenated\n    constraint tokens for that sentence, with 0 appended after\n    each constraint. The first item in each row is the number\n    of constraints for that sentence. So maxlen is the maximum\n    of\n\n    (number of constraints) + (sum length of constraints) + 1.\n\n    across all sentences in the batch.\n    '
    max_constraints_len = 1
    for sentence_constraints in batch_constraints:
        if len(sentence_constraints):
            constraints_len = 1 + sum([c.size(0) for c in sentence_constraints]) + len(sentence_constraints)
            max_constraints_len = max(max_constraints_len, constraints_len)
    batch_size = len(batch_constraints)
    constraints_tensor = torch.zeros((batch_size, max_constraints_len)).long()
    for (i, sentence_constraints) in enumerate(batch_constraints):
        constraints_tensor[i, 0] = len(sentence_constraints)
        offset = 1
        for (j, constraint) in enumerate(sentence_constraints):
            this_len = constraint.size(0)
            constraints_tensor[i, offset:offset + this_len] = constraint
            offset += this_len + 1
    return constraints_tensor.long()

def unpack_constraints(constraint_tensor: torch.Tensor) -> List[torch.Tensor]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Transforms *one row* of a packed constraint tensor (e.g., for one\n    sentence in the batch) into a list of constraint tensors.\n    '
    constraint_list = []
    num_constraints = constraint_tensor[0]
    constraints = constraint_tensor.tolist()
    offset = 1
    for i in range(num_constraints):
        where = constraints.index(0, offset)
        constraint_list.append(constraint_tensor[offset:where])
        offset = where + 1
    return constraint_list

class ConstraintNode:
    """
    Represents a node in a trie managing unordered constraints.
    """

    def __init__(self, token: int=None, parent=None):
        if False:
            for i in range(10):
                print('nop')
        self.token = int(token) if token is not None else None
        self.parent = parent
        self.terminal = 0
        self.children = {}
        self.num_constraints = 0

    @property
    def id(self):
        if False:
            return 10
        return self.token

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        term = self.terminal != 0
        return f'[{self.token}].{term}#{self.num_constraints}'

    def __getitem__(self, key: int):
        if False:
            return 10
        return self.children.get(key, None)

    def next_tokens(self) -> Set[int]:
        if False:
            for i in range(10):
                print('nop')
        'The set of child labels.'
        return set(self.children.keys())

    @staticmethod
    def create(constraints: List[List[int]]):
        if False:
            while True:
                i = 10
        root = ConstraintNode()
        for sequence in constraints:
            root.add_sequence(sequence)
        return root

    @staticmethod
    def print_graph(node: 'ConstraintNode'):
        if False:
            print('Hello World!')
        if len(node.children) == 0:
            return str(node)
        else:
            s = f'({node}'
            for child in node.children.values():
                s += ' ' + ConstraintNode.print_graph(child)
            s += ')'
            return s

    def token_counts(self) -> Counter:
        if False:
            for i in range(10):
                print('nop')
        'Returns a counter of the number of times each token is used\n        in a constraint.\n        '
        token_counts = Counter()
        kids = list(self.children.values())
        while len(kids) > 0:
            kid = kids.pop()
            token_counts[kid.id] += kid.num_constraints
            kids += list(kid.children.values())
        return token_counts

    def tokens(self) -> Set[int]:
        if False:
            i = 10
            return i + 15
        'Returns the set of tokens in constraints.'
        return set(self.token_counts().keys())

    def add_sequence(self, sequence: List[int]):
        if False:
            return 10
        'Adds a constraint, represented as a list of integers, to\n        the trie.'
        assert len(sequence) > 0
        token = int(sequence[0])
        if token not in self.children:
            self.children[token] = ConstraintNode(token, parent=self)
        node = self.children[token]
        if len(sequence) == 1:
            node.terminal += 1
            node.num_constraints += 1
            parent = node.parent
            while parent is not None:
                parent.num_constraints += 1
                parent = parent.parent
        else:
            node.add_sequence(sequence[1:])

class UnorderedConstraintState(ConstraintState):
    """
    Records progress through the set of constraints for each item in the beam
    using a trie.
    """

    def __init__(self, node: ConstraintNode, copy_from: 'ConstraintState'=None):
        if False:
            while True:
                i = 10
        self.node = node
        if copy_from is None:
            self.root = node
            self.completed = Counter()
            self.generated = Counter()
            self.needed_tokens = self.root.tokens()
        else:
            self.completed = Counter(copy_from.completed)
            self.generated = Counter(copy_from.generated)
            self.root = copy_from.root
        if self.node != self.root:
            self.generated[node] += 1

    @staticmethod
    def create(constraint_tensor: torch.Tensor):
        if False:
            print('Hello World!')
        constraint_list = unpack_constraints(constraint_tensor)
        constraint_trie_root = ConstraintNode.create(constraint_list)
        return UnorderedConstraintState(constraint_trie_root)

    def __str__(self):
        if False:
            return 10
        gen_str = ','.join([str(node) for node in self.generated])
        return f'{self.name}/{self.bank}({gen_str})x{self.num_completed}'

    def __copy__(self):
        if False:
            i = 10
            return i + 15
        copied_state = UnorderedConstraintState(self.node, copy_from=self)
        return copied_state

    def copy(self):
        if False:
            i = 10
            return i + 15
        return self.__copy__()

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        if self.node.id is None:
            return 'ROOT'
        else:
            return str(self.node.id)

    @property
    def is_root(self):
        if False:
            print('Hello World!')
        return self.node == self.root

    @property
    def bank(self):
        if False:
            while True:
                i = 10
        return sum(self.generated.values())

    @property
    def num_completed(self):
        if False:
            print('Hello World!')
        'The number of constraints (not constraint tokens) that are completed.\n        In addition to the already-completed states, we need to account for the\n        current state, which might get marked as completed when another token\n        is generated.\n        '
        in_final = self.node.terminal and self.completed[self.node] < self.node.terminal
        return sum(self.completed.values()) + in_final

    @property
    def finished(self):
        if False:
            i = 10
            return i + 15
        return self.root.num_constraints - self.num_completed == 0

    @property
    def token_counts(self):
        if False:
            return 10
        return self.root.token_counts()

    @property
    def tokens(self):
        if False:
            while True:
                i = 10
        return self.root.tokens()

    @property
    def num_constraint_tokens(self):
        if False:
            return 10
        return sum(self.token_counts.values())

    def next_tokens(self) -> Set[int]:
        if False:
            i = 10
            return i + 15
        'Returns the list of tokens that could come next.\n        These are (a) all tokens extending the root state and, for\n        non-root states, additionally all tokens extending the current\n        state.'
        if self.node != self.root:
            return self.root.next_tokens().union(self.node.next_tokens())
        else:
            return self.root.next_tokens()

    def advance(self, token: int):
        if False:
            print('Hello World!')
        'Reads in a token and advances the state. Here\'s how it works.\n\n        We can advance to the next state if:\n        - there is a matching child\n        - its path isn\'t blocked\n\n        A path is blocked when all constraints that are descendants of\n        that node have already been generated, in the current state.\n\n        If we are not able to advance from the current state, we "fall\n        off the graph" and return to the root state. There, we again\n        try to advance, checking the same criteria.\n\n        In any case, when falling off the graph, we need to do some\n        bookkeeping. We:\n        - check whether any constraints were met (all prefixes of\n          current state)\n        - if one is found, mark it as completed\n        - adjust visited nodes accordingly\n        '
        token = int(token)
        next_state = None
        child = self.node[token]
        if child is not None and self.generated[child] < child.num_constraints:
            next_state = UnorderedConstraintState(child, copy_from=self)

        def rewind():
            if False:
                for i in range(10):
                    print('nop')
            'If we\'re mid-trie and an "illegal" token is chosen next, we need\n            to reset our state to the root state. However, along the way, we need\n            to check whether a prefix of the current trie state represents a state\n            we could mark as completed.\n            '
            node = self.node
            while node != self.root:
                if node.terminal and self.completed[node] < node.terminal:
                    next_state.completed[node] += 1
                    return
                next_state.generated[node] -= 1
                node = node.parent
        if next_state is None and token in self.root.next_tokens():
            child = self.root[token]
            if self.generated[child] < child.num_constraints:
                next_state = UnorderedConstraintState(child, copy_from=self)
            else:
                next_state = UnorderedConstraintState(self.root, copy_from=self)
            rewind()
        elif next_state is None:
            next_state = UnorderedConstraintState(self.root, copy_from=self)
            rewind()
        return next_state

class ConstraintSequence:

    def __init__(self, sequences: List[List[int]]):
        if False:
            while True:
                i = 10
        'Represents a set of possibly multitoken constraints by\n        concatenating them and internally recording the end points.\n        '
        self.sequences = []
        self.endpoints = []
        self.num_tokens = 0
        self.tokens = set()
        for sequence in sequences:
            for token in sequence:
                self.tokens.add(token)
            self.num_tokens += len(sequence)
            self.endpoints += [False for x in range(len(sequence) - 1)] + [True]
            self.sequences += sequence

    def __getitem__(self, key: int):
        if False:
            while True:
                i = 10
        return self.sequences[key]

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self.sequences)

    def __str__(self):
        if False:
            return 10
        return str(self.sequences)

class OrderedConstraintState(ConstraintState):
    """
    Records progress through the set of linear nonbranching constraints with gaps.
    """

    def __init__(self, sequence: ConstraintSequence, state: int=-1):
        if False:
            return 10
        self.sequence = sequence
        self.state = state

    @staticmethod
    def create(constraint_tensor: torch.Tensor):
        if False:
            i = 10
            return i + 15
        constraint_list = unpack_constraints(constraint_tensor)
        return OrderedConstraintState(ConstraintSequence(constraint_list), -1)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return f'{self.state}/{self.bank}x{self.num_completed}'

    def __copy__(self):
        if False:
            i = 10
            return i + 15
        return OrderedConstraintState(self.sequence, self.state)

    def copy(self):
        if False:
            print('Hello World!')
        return self.__copy__()

    @property
    def num_completed(self):
        if False:
            i = 10
            return i + 15
        if self.state == -1:
            return 0
        count = len(list(filter(lambda x: x, self.sequence.endpoints[0:self.state + 1])))
        return count

    @property
    def is_root(self):
        if False:
            i = 10
            return i + 15
        return self.state == -1

    @property
    def name(self):
        if False:
            return 10
        if self.state == -1:
            return 'ROOT'
        else:
            return str(self.sequence[self.state])

    @property
    def bank(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.state + 1

    @property
    def finished(self):
        if False:
            i = 10
            return i + 15
        return self.state + 1 == len(self.sequence)

    @property
    def token_counts(self):
        if False:
            for i in range(10):
                print('nop')
        return self.sequence.token_counts()

    @property
    def tokens(self):
        if False:
            return 10
        return self.sequence.tokens

    @property
    def num_constraint_tokens(self):
        if False:
            for i in range(10):
                print('nop')
        return sum(self.token_counts.values())

    def next_tokens(self) -> Set[int]:
        if False:
            while True:
                i = 10
        'Returns the list of tokens that could come next.\n        These are (a) all tokens extending the root state and, for\n        non-root states, additionally all tokens extending the current\n        state.'
        tokens = set()
        if self.state > 0:
            tokens.add(self.sequence[0])
        if not self.finished:
            tokens.add(self.sequence[self.state + 1])
        return tokens

    def advance(self, token: int):
        if False:
            return 10
        'Reads in a token and advances the state. Here\'s how it works.\n\n        We can advance to the next state if:\n        - there is a matching child\n        - its path isn\'t blocked\n\n        A path is blocked when all constraints that are descendants of\n        that node have already been generated, in the current state.\n\n        If we are not able to advance from the current state, we "fall\n        off the graph" and return to the root state. There, we again\n        try to advance, checking the same criteria.\n\n        In any case, when falling off the graph, we need to do some\n        bookkeeping. We:\n        - check whether any constraints were met (all prefixes of\n          current state)\n        - if one is found, mark it as completed\n        - adjust visited nodes accordingly\n        '
        token = int(token)
        if self.finished:
            next_state = self.copy()
        elif self.sequence[self.state + 1] == token:
            next_state = OrderedConstraintState(self.sequence, self.state + 1)
        elif self.sequence.endpoints[self.state]:
            next_state = self.copy()
        elif token == self.sequence[0]:
            next_state = OrderedConstraintState(self.sequence, 0)
        else:
            next_state = OrderedConstraintState(self.sequence, -1)
        return next_state