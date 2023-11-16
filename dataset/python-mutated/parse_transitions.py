"""
Defines a series of transitions (open a constituent, close a constituent, etc

Also defines a State which holds the various data needed to build
a parse tree out of tagged words.
"""
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from enum import Enum
import functools
import logging
from stanza.models.constituency.parse_tree import Tree
logger = logging.getLogger('stanza')

class TransitionScheme(Enum):
    TOP_DOWN = 1
    TOP_DOWN_COMPOUND = 2
    TOP_DOWN_UNARY = 3
    IN_ORDER = 4
    IN_ORDER_COMPOUND = 5
    IN_ORDER_UNARY = 6

class State(namedtuple('State', ['word_queue', 'transitions', 'constituents', 'gold_tree', 'gold_sequence', 'sentence_length', 'num_opens', 'word_position', 'score'])):
    """
    Represents a partially completed transition parse

    Includes stack/buffers for unused words, already executed transitions, and partially build constituents
    At training time, also keeps track of the gold data we are reparsing

    num_opens is useful for tracking
       1) if the parser is in a stuck state where it is making infinite opens
       2) if a close transition is impossible because there are no previous opens

    sentence_length tracks how long the sentence is so we abort if we go infinite

    non-stack information such as sentence_length and num_opens
    will be copied from the original_state if possible, with the
    exact arguments overriding the values in the original_state

    gold_tree: the original tree, if made from a gold tree.  might be None
    gold_sequence: the original transition sequence, if available
    Note that at runtime, gold values will not be available

    word_position tracks where in the word queue we are.  cheaper than
      manipulating the list itself.  this can be handled differently
      from transitions and constituents as it is processed once
      at the start of parsing

    The word_queue should have both a start and an end word.
    Those can be None in the case of the endpoints if they are unused.
    """

    def empty_word_queue(self):
        if False:
            i = 10
            return i + 15
        return self.word_position == self.sentence_length

    def empty_transitions(self):
        if False:
            for i in range(10):
                print('nop')
        return self.transitions.parent is None

    def has_one_constituent(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.constituents) == 2

    def num_constituents(self):
        if False:
            i = 10
            return i + 15
        return len(self.constituents) - 1

    def num_transitions(self):
        if False:
            i = 10
            return i + 15
        return len(self.transitions) - 1

    def get_word(self, pos):
        if False:
            for i in range(10):
                print('nop')
        return self.word_queue[pos + 1]

    def finished(self, model):
        if False:
            for i in range(10):
                print('nop')
        return self.empty_word_queue() and self.has_one_constituent() and (model.get_top_constituent(self.constituents).label in model.get_root_labels())

    def get_tree(self, model):
        if False:
            while True:
                i = 10
        return model.get_top_constituent(self.constituents)

    def all_transitions(self, model):
        if False:
            for i in range(10):
                print('nop')
        all_transitions = []
        transitions = self.transitions
        while transitions.parent is not None:
            all_transitions.append(model.get_top_transition(transitions))
            transitions = transitions.parent
        return list(reversed(all_transitions))

    def all_constituents(self, model):
        if False:
            for i in range(10):
                print('nop')
        all_constituents = []
        constituents = self.constituents
        while constituents.parent is not None:
            all_constituents.append(model.get_top_constituent(constituents))
            constituents = constituents.parent
        return list(reversed(all_constituents))

    def all_words(self, model):
        if False:
            return 10
        return [model.get_word(x) for x in self.word_queue]

    def to_string(self, model):
        if False:
            i = 10
            return i + 15
        return 'State(\n  buffer:%s\n  transitions:%s\n  constituents:%s\n  word_position:%d num_opens:%d)' % (str(self.all_words(model)), str(self.all_transitions(model)), str(self.all_constituents(model)), self.word_position, self.num_opens)

    def __str__(self):
        if False:
            while True:
                i = 10
        return 'State(\n  buffer:%s\n  transitions:%s\n  constituents:%s)' % (str(self.word_queue), str(self.transitions), str(self.constituents))

@functools.total_ordering
class Transition(ABC):
    """
    model is passed in as a dependency injection
    for example, an LSTM model can update hidden & output vectors when transitioning
    """

    @abstractmethod
    def update_state(self, state, model):
        if False:
            while True:
                i = 10
        "\n        update the word queue position, possibly remove old pieces from the constituents state, and return the new constituent\n\n        the return value should be a tuple:\n          updated word_position\n          updated constituents\n          new constituent to put on the queue and None\n            - note that the constituent shouldn't be on the queue yet\n              that allows putting it on as a batch operation, which\n              saves a significant amount of time in an LSTM, for example\n          OR\n          data used to make a new constituent and the method used\n            - for example, CloseConstituent can return the children needed\n              and itself.  this allows a batch operation to build\n              the constituent\n        "

    def delta_opens(self):
        if False:
            return 10
        return 0

    def apply(self, state, model):
        if False:
            print('Hello World!')
        '\n        return a new State transformed via this transition\n\n        convenience method to call bulk_apply, which is significantly\n        faster than single operations for an NN based model\n        '
        update = bulk_apply(model, [state], [self])
        return update[0]

    @abstractmethod
    def is_legal(self, state, model):
        if False:
            i = 10
            return i + 15
        '\n        assess whether or not this transition is legal in this state\n\n        at parse time, the parser might choose a transition which cannot be made\n        '

    def components(self):
        if False:
            i = 10
            return i + 15
        '\n        Return a list of transitions which could theoretically make up this transition\n\n        For example, an Open transition with multiple labels would\n        return a list of Opens with those labels\n        '
        return [self]

    @abstractmethod
    def short_name(self):
        if False:
            while True:
                i = 10
        '\n        A short name to identify this transition\n        '

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        if self == other:
            return False
        if isinstance(self, Shift):
            return True
        if isinstance(other, Shift):
            return False
        return str(self) < str(other)

class Shift(Transition):

    def update_state(self, state, model):
        if False:
            for i in range(10):
                print('nop')
        '\n        This will handle all aspects of a shift transition\n\n        - push the top element of the word queue onto constituents\n        - pop the top element of the word queue\n        '
        new_constituent = model.transform_word_to_constituent(state)
        return (state.word_position + 1, state.constituents, new_constituent, None)

    def is_legal(self, state, model):
        if False:
            print('Hello World!')
        '\n        Disallow shifting when the word queue is empty or there are no opens to eventually eat this word\n        '
        if state.empty_word_queue():
            return False
        if model.is_top_down():
            if state.num_opens == 0:
                return False
            if state.num_opens == 1:
                assert state.transitions.parent is not None
                if state.transitions.parent.parent is None:
                    trans = model.get_top_transition(state.transitions)
                    if len(trans.label) == 1 and trans.top_label in model.get_root_labels():
                        return False
        elif state.num_opens == 0:
            if state.num_constituents() > 0:
                return False
        return True

    def short_name(self):
        if False:
            i = 10
            return i + 15
        return 'Shift'

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'Shift'

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if self is other:
            return True
        if isinstance(other, Shift):
            return True
        return False

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash(37)

class CompoundUnary(Transition):

    def __init__(self, *label):
        if False:
            return 10
        self.label = tuple(label)

    def update_state(self, state, model):
        if False:
            while True:
                i = 10
        '\n        Apply potentially multiple unary transitions to the same preterminal\n\n        It reuses the CloseConstituent machinery\n        '
        constituents = state.constituents
        children = [constituents.value]
        constituents = constituents.pop()
        return (state.word_position, constituents, (self.label, children), CloseConstituent)

    def is_legal(self, state, model):
        if False:
            return 10
        '\n        Disallow consecutive CompoundUnary transitions, force final transition to go to ROOT\n        '
        tree = model.get_top_constituent(state.constituents)
        if tree is None:
            return False
        if isinstance(model.get_top_transition(state.transitions), (CompoundUnary, OpenConstituent)):
            return False
        if model.transition_scheme() is TransitionScheme.IN_ORDER_COMPOUND:
            return tree.is_preterminal()
        if model.transition_scheme() is not TransitionScheme.TOP_DOWN_UNARY:
            return True
        is_root = self.label[0] in model.get_root_labels()
        if not state.empty_word_queue() or not state.has_one_constituent():
            return not is_root
        else:
            return is_root

    def components(self):
        if False:
            print('Hello World!')
        return [CompoundUnary(label) for label in self.label]

    def short_name(self):
        if False:
            return 10
        return 'Unary'

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'CompoundUnary(%s)' % ','.join(self.label)

    def __eq__(self, other):
        if False:
            return 10
        if self is other:
            return True
        if not isinstance(other, CompoundUnary):
            return False
        if self.label == other.label:
            return True
        return False

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash(self.label)

class Dummy:
    """
    Takes a space on the constituent stack to represent where an Open transition occurred
    """

    def __init__(self, label):
        if False:
            while True:
                i = 10
        self.label = label

    def is_preterminal(self):
        if False:
            i = 10
            return i + 15
        return False

    def __str__(self):
        if False:
            return 10
        return 'Dummy({})'.format(self.label)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if self is other:
            return True
        if not isinstance(other, Dummy):
            return False
        if self.label == other.label:
            return True
        return False

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash(self.label)

def too_many_unary_nodes(tree, unary_limit):
    if False:
        i = 10
        return i + 15
    '\n    Return True iff there are UNARY_LIMIT unary nodes in a tree in a row\n\n    helps prevent infinite open/close patterns\n    otherwise, the model can get stuck in essentially an infinite loop\n    '
    if tree is None:
        return False
    for _ in range(unary_limit + 1):
        if len(tree.children) != 1:
            return False
        tree = tree.children[0]
    return True

class OpenConstituent(Transition):

    def __init__(self, *label):
        if False:
            i = 10
            return i + 15
        self.label = tuple(label)
        self.top_label = self.label[0]

    def delta_opens(self):
        if False:
            while True:
                i = 10
        return 1

    def update_state(self, state, model):
        if False:
            while True:
                i = 10
        return (state.word_position, state.constituents, model.dummy_constituent(Dummy(self.label)), None)

    def is_legal(self, state, model):
        if False:
            i = 10
            return i + 15
        '\n        disallow based on the length of the sentence\n        '
        if state.num_opens > state.sentence_length + 10:
            return False
        if model.is_top_down():
            if state.empty_word_queue():
                return False
            if not model.has_unary_transitions():
                is_root = self.top_label in model.get_root_labels()
                if is_root:
                    return state.empty_transitions()
                else:
                    return not state.empty_transitions()
        else:
            if state.num_constituents() == 0:
                return False
            if isinstance(model.get_top_transition(state.transitions), OpenConstituent):
                return False
            if model.transition_scheme() is TransitionScheme.IN_ORDER_UNARY or model.transition_scheme() is TransitionScheme.IN_ORDER_COMPOUND:
                return not state.empty_word_queue()
            is_root = self.top_label in model.get_root_labels()
            if is_root:
                return state.num_opens == 0 and state.empty_word_queue()
            else:
                if (state.num_opens > 0 or state.empty_word_queue()) and too_many_unary_nodes(model.get_top_constituent(state.constituents), model.unary_limit()):
                    return False
                return True
        return True

    def components(self):
        if False:
            print('Hello World!')
        return [OpenConstituent(label) for label in self.label]

    def short_name(self):
        if False:
            i = 10
            return i + 15
        return 'Open'

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'OpenConstituent({})'.format(self.label)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if self is other:
            return True
        if not isinstance(other, OpenConstituent):
            return False
        if self.label == other.label:
            return True
        return False

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash(self.label)

class Finalize(Transition):
    """
    Specifically applies at the end of a parse sequence to add a ROOT

    Seemed like the simplest way to remove ROOT from the
    in_order_compound transitions while still using the mechanism of
    the transitions to build the parse tree
    """

    def __init__(self, *label):
        if False:
            return 10
        self.label = tuple(label)

    def update_state(self, state, model):
        if False:
            while True:
                i = 10
        '\n        Apply potentially multiple unary transitions to the same preterminal\n\n        Only applies to preterminals\n        It reuses the CloseConstituent machinery\n        '
        constituents = state.constituents
        children = [constituents.value]
        constituents = constituents.pop()
        label = self.label
        return (state.word_position, constituents, (label, children), CloseConstituent)

    def is_legal(self, state, model):
        if False:
            print('Hello World!')
        '\n        Legal if & only if there is one tree, no more words, and no ROOT yet\n        '
        return state.empty_word_queue() and state.has_one_constituent() and (not state.finished(model))

    def short_name(self):
        if False:
            print('Hello World!')
        return 'Finalize'

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'Finalize(%s)' % ','.join(self.label)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if self is other:
            return True
        if not isinstance(other, Finalize):
            return False
        return other.label == self.label

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash((53, self.label))

class CloseConstituent(Transition):

    def delta_opens(self):
        if False:
            return 10
        return -1

    def update_state(self, state, model):
        if False:
            while True:
                i = 10
        children = []
        constituents = state.constituents
        while not isinstance(model.get_top_constituent(constituents), Dummy):
            children.append(constituents.value)
            constituents = constituents.pop()
        label = model.get_top_constituent(constituents).label
        constituents = constituents.pop()
        if not model.is_top_down():
            children.append(constituents.value)
            constituents = constituents.pop()
        children.reverse()
        return (state.word_position, constituents, (label, children), CloseConstituent)

    @staticmethod
    def build_constituents(model, data):
        if False:
            return 10
        '\n        builds new constituents out of the incoming data\n\n        data is a list of tuples: (label, children)\n        the model will batch the build operation\n        again, the purpose of this batching is to do multiple deep learning operations at once\n        '
        (labels, children_lists) = map(list, zip(*data))
        new_constituents = model.build_constituents(labels, children_lists)
        return new_constituents

    def is_legal(self, state, model):
        if False:
            for i in range(10):
                print('nop')
        '\n        Disallow if there is no Open on the stack yet\n\n        in TOP_DOWN, if the previous transition was the Open (nothing built yet)\n        in IN_ORDER, previous transition does not matter, except for one small corner case\n        '
        if state.num_opens <= 0:
            return False
        if model.is_top_down():
            if isinstance(model.get_top_transition(state.transitions), OpenConstituent):
                return False
            if state.num_opens <= 1 and (not state.empty_word_queue()):
                return False
            if model.transition_scheme() == TransitionScheme.TOP_DOWN_COMPOUND:
                if state.num_opens == 1 and (not state.empty_word_queue()):
                    return False
            elif not model.has_unary_transitions():
                if state.num_opens == 2 and (not state.empty_word_queue()):
                    return False
        elif model.transition_scheme() == TransitionScheme.IN_ORDER:
            if not isinstance(model.get_top_transition(state.transitions), OpenConstituent):
                return True
            if isinstance(model.get_top_transition(state.transitions), OpenConstituent) and (model.transition_scheme() is TransitionScheme.IN_ORDER_UNARY or model.transition_scheme() is TransitionScheme.IN_ORDER_COMPOUND):
                return False
            if state.num_opens > 1 or state.empty_word_queue():
                return True
            node = model.get_top_constituent(state.constituents.pop())
            if too_many_unary_nodes(node, model.unary_limit()):
                return False
        elif model.transition_scheme() == TransitionScheme.IN_ORDER_COMPOUND:
            if isinstance(model.get_top_transition(state.transitions), OpenConstituent):
                return False
        return True

    def short_name(self):
        if False:
            i = 10
            return i + 15
        return 'Close'

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'CloseConstituent'

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if self is other:
            return True
        if isinstance(other, CloseConstituent):
            return True
        return False

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(93)

def check_transitions(train_transitions, other_transitions, treebank_name):
    if False:
        i = 10
        return i + 15
    "\n    Check that all the transitions in the other dataset are known in the train set\n\n    Weird nested unaries are warned rather than failed as long as the\n    components are all known\n\n    There is a tree in VLSP, for example, with three (!) nested NP nodes\n    If this is an unknown compound transition, we won't possibly get it\n    right when parsing, but at least we don't need to fail\n    "
    unknown_transitions = set()
    for trans in other_transitions:
        if trans not in train_transitions:
            for component in trans.components():
                if component not in train_transitions:
                    raise RuntimeError("Found transition {} in the {} set which don't exist in the train set".format(trans, treebank_name))
            unknown_transitions.add(trans)
    if len(unknown_transitions) > 0:
        logger.warning('Found transitions where the components are all valid transitions, but the complete transition is unknown: %s', sorted(unknown_transitions))

def bulk_apply(model, state_batch, transitions, fail=False):
    if False:
        while True:
            i = 10
    '\n    Apply the given list of Transitions to the given list of States, using the model as a reference\n\n    model: SimpleModel, LSTMModel, or any other form of model\n    state_batch: list of States\n    transitions: list of transitions, one per state\n    fail: throw an exception on a failed transition, as opposed to skipping the tree\n    '
    remove = set()
    word_positions = []
    constituents = []
    new_constituents = []
    callbacks = defaultdict(list)
    for (idx, (tree, transition)) in enumerate(zip(state_batch, transitions)):
        if not transition:
            error = "Got stuck and couldn't find a legal transition on the following gold tree:\n{}\n\nFinal state:\n{}".format(tree.gold_tree, tree.to_string(model))
            if fail:
                raise ValueError(error)
            else:
                logger.error(error)
                remove.add(idx)
                continue
        if tree.num_transitions() >= len(tree.word_queue) * 20:
            if tree.gold_tree:
                error = 'Went infinite on the following gold tree:\n{}\n\nFinal state:\n{}'.format(tree.gold_tree, tree.to_string(model))
            else:
                error = 'Went infinite!:\nFinal state:\n{}'.format(tree.to_string(model))
            if fail:
                raise ValueError(error)
            else:
                logger.error(error)
                remove.add(idx)
                continue
        (wq, c, nc, callback) = transition.update_state(tree, model)
        word_positions.append(wq)
        constituents.append(c)
        new_constituents.append(nc)
        if callback:
            callbacks[callback].append(len(new_constituents) - 1)
    for (key, idxs) in callbacks.items():
        data = [new_constituents[x] for x in idxs]
        callback_constituents = key.build_constituents(model, data)
        for (idx, constituent) in zip(idxs, callback_constituents):
            new_constituents[idx] = constituent
    state_batch = [tree for (idx, tree) in enumerate(state_batch) if idx not in remove]
    transitions = [trans for (idx, trans) in enumerate(transitions) if idx not in remove]
    if len(state_batch) == 0:
        return state_batch
    new_transitions = model.push_transitions([tree.transitions for tree in state_batch], transitions)
    new_constituents = model.push_constituents(constituents, new_constituents)
    state_batch = [state._replace(num_opens=state.num_opens + transition.delta_opens(), word_position=word_position, transitions=transition_stack, constituents=constituents) for (state, transition, word_position, transition_stack, constituents) in zip(state_batch, transitions, word_positions, new_transitions, new_constituents)]
    return state_batch