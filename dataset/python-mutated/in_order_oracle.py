from enum import Enum, auto
from stanza.models.constituency.dynamic_oracle import DynamicOracle
from stanza.models.constituency.parse_transitions import Shift, OpenConstituent, CloseConstituent

def fix_wrong_open_root_error(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    if False:
        return 10
    '\n    If there is an open/open error specifically at the ROOT, close the wrong open and try again\n    '
    if gold_transition == pred_transition:
        return None
    if isinstance(gold_transition, OpenConstituent) and isinstance(pred_transition, OpenConstituent) and (gold_transition.top_label in root_labels):
        return gold_sequence[:gold_index] + [pred_transition, CloseConstituent()] + gold_sequence[gold_index:]
    return None

def fix_wrong_open_unary_chain(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    if False:
        while True:
            i = 10
    '\n    Fix a wrong open/open in a unary chain by removing the skipped unary transitions\n\n    Only applies is the wrong pred transition is a transition found higher up in the unary chain\n    '
    if gold_transition == pred_transition:
        return None
    if isinstance(gold_transition, OpenConstituent) and isinstance(pred_transition, OpenConstituent):
        cur_index = gold_index + 1
        while cur_index + 1 < len(gold_sequence) and isinstance(gold_sequence[cur_index], CloseConstituent) and isinstance(gold_sequence[cur_index + 1], OpenConstituent):
            cur_index = cur_index + 1
            if gold_sequence[cur_index] == pred_transition:
                return gold_sequence[:gold_index] + gold_sequence[cur_index:]
            cur_index = cur_index + 1
    return None

def advance_past_constituents(gold_sequence, cur_index):
    if False:
        for i in range(10):
            print('nop')
    '\n    Advance cur_index through gold_sequence until we have seen 1 more Close than Open\n\n    The index returned is the index of the Close which occurred after all the stuff\n    '
    count = 0
    while cur_index < len(gold_sequence):
        if isinstance(gold_sequence[cur_index], OpenConstituent):
            count = count + 1
        elif isinstance(gold_sequence[cur_index], CloseConstituent):
            count = count - 1
            if count == -1:
                return cur_index
        cur_index = cur_index + 1
    return None

def find_constituent_end(gold_sequence, cur_index):
    if False:
        i = 10
        return i + 15
    '\n    Advance cur_index through gold_sequence until the next block has ended\n\n    This is different from advance_past_constituents in that it will\n    also return when there is a Shift when count == 0.  That way, we\n    return the first block of things we know attach to the left\n    '
    count = 0
    saw_shift = False
    while cur_index < len(gold_sequence):
        if isinstance(gold_sequence[cur_index], OpenConstituent):
            count = count + 1
        elif isinstance(gold_sequence[cur_index], CloseConstituent):
            count = count - 1
            if count == -1:
                return cur_index
        elif isinstance(gold_sequence[cur_index], Shift):
            if saw_shift and count == 0:
                return cur_index
            else:
                saw_shift = True
        cur_index = cur_index + 1
    return None

def advance_past_unaries(gold_sequence, cur_index):
    if False:
        print('Hello World!')
    while cur_index + 2 < len(gold_sequence) and isinstance(gold_sequence[cur_index], OpenConstituent) and isinstance(gold_sequence[cur_index + 1], CloseConstituent):
        cur_index += 2
    return cur_index

def fix_wrong_open_stuff_unary(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    if False:
        while True:
            i = 10
    '\n    Fix a wrong open/open when there is an intervening constituent and then the guessed NT\n\n    This happens when the correct pattern is\n      stuff_1 NT_X stuff_2 close NT_Y ...\n    and instead of guessing the gold transition NT_X,\n    the prediction was NT_Y\n    '
    if gold_transition == pred_transition:
        return None
    if not isinstance(gold_transition, OpenConstituent):
        return None
    if not isinstance(pred_transition, OpenConstituent):
        return None
    stuff_start = gold_index + 1
    if not isinstance(gold_sequence[stuff_start], Shift):
        return None
    stuff_end = advance_past_constituents(gold_sequence, stuff_start)
    if stuff_end is None:
        return None
    cur_index = stuff_end + 1
    while isinstance(gold_sequence[cur_index], OpenConstituent):
        if gold_sequence[cur_index] == pred_transition:
            return gold_sequence[:gold_index] + [pred_transition] + gold_sequence[stuff_start:stuff_end] + gold_sequence[cur_index + 1:]
        if cur_index + 2 < len(gold_sequence) and isinstance(gold_sequence[cur_index + 1], CloseConstituent):
            cur_index = cur_index + 2
        else:
            break
    return None

def fix_wrong_open_general(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    if False:
        return 10
    '\n    Fix a general wrong open/open transition by accepting the open and continuing\n\n    A couple other open/open patterns have already been carved out\n    '
    if gold_transition == pred_transition:
        return None
    if not isinstance(gold_transition, OpenConstituent):
        return None
    if not isinstance(pred_transition, OpenConstituent):
        return None
    if gold_transition.top_label in root_labels:
        return None
    return gold_sequence[:gold_index] + [pred_transition] + gold_sequence[gold_index + 1:]

def fix_missed_unary(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    if False:
        i = 10
        return i + 15
    '\n    Fix a missed unary which is followed by an otherwise correct transition\n\n    (also handles multiple missed unary transitions)\n    '
    if gold_transition == pred_transition:
        return None
    cur_index = gold_index
    cur_index = advance_past_unaries(gold_sequence, cur_index)
    if gold_sequence[cur_index] == pred_transition:
        return gold_sequence[:gold_index] + gold_sequence[cur_index:]
    return None

def fix_open_shift(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    if False:
        i = 10
        return i + 15
    '\n    Fix an Open replaced with a Shift\n\n    Suppose we were supposed to guess NT_X and instead did S\n\n    We derive the repair as follows.\n\n    For simplicity, assume the open is not a unary for now\n\n    Since we know an Open was legal, there must be stuff\n      stuff NT_X\n    Shift is also legal, so there must be other stuff and a previous Open\n      stuff_1 NT_Y stuff_2 NT_X\n    After the NT_X which we missed, there was a bunch of stuff and a close for NT_X\n      stuff_1 NT_Y stuff_2 NT_X stuff_3 C\n    There could be more stuff here which can be saved...\n      stuff_1 NT_Y stuff_2 NT_X stuff_3 C stuff_4 C\n      stuff_1 NT_Y stuff_2 NT_X stuff_3 C C\n    '
    if not isinstance(gold_transition, OpenConstituent):
        return None
    if not isinstance(pred_transition, Shift):
        return None
    cur_index = gold_index
    cur_index = advance_past_unaries(gold_sequence, cur_index)
    if not isinstance(gold_sequence[cur_index], OpenConstituent):
        return None
    if gold_sequence[cur_index].top_label in root_labels:
        return None
    stuff_start = cur_index + 1
    assert isinstance(gold_sequence[stuff_start], Shift)
    stuff_end = advance_past_constituents(gold_sequence, stuff_start)
    cur_index = stuff_end + 1
    if cur_index >= len(gold_sequence):
        return None
    if isinstance(gold_sequence[cur_index], OpenConstituent):
        cur_index = advance_past_unaries(gold_sequence, cur_index)
        if cur_index >= len(gold_sequence):
            return None
    if isinstance(gold_sequence[cur_index], OpenConstituent):
        return None
    repair = gold_sequence[:gold_index] + gold_sequence[stuff_start:stuff_end] + gold_sequence[cur_index:]
    return repair

def find_previous_open(gold_sequence, cur_index):
    if False:
        print('Hello World!')
    "\n    Go backwards from cur_index to find the open which opens the previous block of stuff.\n\n    Return None if it can't be found.\n    "
    count = 0
    cur_index = cur_index - 1
    while cur_index >= 0:
        if isinstance(gold_sequence[cur_index], OpenConstituent):
            count = count + 1
            if count > 0:
                return cur_index
        elif isinstance(gold_sequence[cur_index], CloseConstituent):
            count = count - 1
        cur_index = cur_index - 1
    return None

def fix_open_close(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    if False:
        for i in range(10):
            print('nop')
    '\n    Fix an Open replaced with a Close\n\n    Call the Open NT_X\n    Open legal, so there must be stuff:\n      stuff NT_X\n    Close legal, so there must be something to close:\n      stuff_1 NT_Y stuff_2 NT_X\n\n    The incorrect close makes the following brackets:\n      (Y stuff_1 stuff_2)\n    We were supposed to build\n      (Y stuff_1 (X stuff_2 ...) (possibly more stuff))\n    The simplest fix here is to reopen Y at this point.\n\n    One issue might be if there is another bracket which encloses X underneath Y\n    So, for example, the tree was supposed to be\n      (Y stuff_1 (Z (X stuff_2 stuff_3) stuff_4))\n    The pattern for this case is\n      stuff_1 NT_Y stuff_2 NY_X stuff_3 close NT_Z stuff_4 close close\n    '
    if not isinstance(gold_transition, OpenConstituent):
        return None
    if not isinstance(pred_transition, CloseConstituent):
        return None
    cur_index = advance_past_unaries(gold_sequence, gold_index)
    if cur_index >= len(gold_sequence):
        return None
    if not isinstance(gold_sequence[cur_index], OpenConstituent):
        return None
    if gold_sequence[cur_index].top_label in root_labels:
        return None
    prev_open_index = find_previous_open(gold_sequence, gold_index)
    if prev_open_index is None:
        return None
    prev_open = gold_sequence[prev_open_index]
    stuff_start = cur_index + 1
    assert isinstance(gold_sequence[stuff_start], Shift)
    stuff_end = advance_past_constituents(gold_sequence, stuff_start)
    cur_index = stuff_end + 1
    if cur_index >= len(gold_sequence):
        return None
    cur_index = advance_past_unaries(gold_sequence, cur_index)
    if isinstance(gold_sequence[cur_index], OpenConstituent):
        return None
    repair = gold_sequence[:gold_index] + [pred_transition, prev_open] + gold_sequence[stuff_start:stuff_end] + gold_sequence[cur_index:]
    return repair

def fix_shift_close(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    if False:
        while True:
            i = 10
    '\n    This fixes Shift replaced with a Close transition.\n\n    This error occurs in the following pattern:\n      stuff_1 NT_X stuff... shift\n    Instead of shift, you close the NT_X\n    The easiest fix here is to just restore the NT_X.\n    '
    if not isinstance(pred_transition, CloseConstituent):
        return None
    cur_index = gold_index
    if isinstance(gold_transition, OpenConstituent):
        cur_index = advance_past_unaries(gold_sequence, cur_index)
    if not isinstance(gold_sequence[cur_index], Shift):
        return None
    prev_open_index = find_previous_open(gold_sequence, gold_index)
    if prev_open_index is None:
        return None
    prev_open = gold_sequence[prev_open_index]
    return gold_sequence[:gold_index] + [pred_transition, prev_open] + gold_sequence[cur_index:]

def fix_close_shift_nested(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    if False:
        i = 10
        return i + 15
    '\n    Fix a Close X..Open X..Shift pattern where both the Close and Open were skipped.\n    '
    if not isinstance(gold_transition, CloseConstituent):
        return None
    if not isinstance(pred_transition, Shift):
        return None
    if len(gold_sequence) < gold_index + 3:
        return None
    prev_open_index = find_previous_open(gold_sequence, gold_index)
    if prev_open_index is None:
        return None
    prev_open = gold_sequence[prev_open_index]
    if gold_sequence[gold_index + 1] != prev_open:
        return None
    if not isinstance(gold_sequence[gold_index + 2], Shift):
        return None
    return gold_sequence[:gold_index] + gold_sequence[gold_index + 2:]

def fix_close_shift_shift(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    if False:
        for i in range(10):
            print('nop')
    '\n    Repair Close/Shift -> Shift by moving the Close to after the next block is created\n    '
    if not isinstance(gold_transition, CloseConstituent):
        return None
    if not isinstance(pred_transition, Shift):
        return None
    if len(gold_sequence) < gold_index + 2:
        return None
    start_index = gold_index + 1
    start_index = advance_past_unaries(gold_sequence, start_index)
    if len(gold_sequence) < start_index + 2:
        return None
    if not isinstance(gold_sequence[start_index], Shift):
        return None
    end_index = find_constituent_end(gold_sequence, start_index)
    if end_index is None:
        return None
    if not isinstance(gold_sequence[end_index], CloseConstituent):
        return None
    return gold_sequence[:gold_index] + gold_sequence[start_index:end_index] + [CloseConstituent()] + gold_sequence[end_index:]

class RepairType(Enum):
    """
    Keep track of which repair is used, if any, on an incorrect transition
    """

    def __new__(cls, fn, correct=False):
        if False:
            i = 10
            return i + 15
        '\n        Enumerate values as normal, but also keep a pointer to a function which repairs that kind of error\n        '
        value = len(cls.__members__)
        obj = object.__new__(cls)
        obj._value_ = value + 1
        obj.fn = fn
        obj.correct = correct
        return obj

    def is_correct(self):
        if False:
            i = 10
            return i + 15
        return self.correct
    WRONG_OPEN_ROOT_ERROR = (fix_wrong_open_root_error,)
    WRONG_OPEN_UNARY_CHAIN = (fix_wrong_open_unary_chain,)
    WRONG_OPEN_STUFF_UNARY = (fix_wrong_open_stuff_unary,)
    WRONG_OPEN_GENERAL = (fix_wrong_open_general,)
    MISSED_UNARY = (fix_missed_unary,)
    OPEN_SHIFT = (fix_open_shift,)
    OPEN_CLOSE = (fix_open_close,)
    SHIFT_CLOSE = (fix_shift_close,)
    CLOSE_SHIFT_NESTED = (fix_close_shift_nested,)
    CORRECT = (None, True)
    UNKNOWN = None

class InOrderOracle(DynamicOracle):

    def __init__(self, root_labels, oracle_level):
        if False:
            return 10
        super().__init__(root_labels, oracle_level, RepairType)