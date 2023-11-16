from hypothesis.internal.compat import int_from_bytes, int_to_bytes
from hypothesis.internal.conjecture.data import Status
from hypothesis.internal.conjecture.engine import BUFFER_SIZE
from hypothesis.internal.conjecture.junkdrawer import find_integer
from hypothesis.internal.conjecture.pareto import NO_SCORE

class Optimiser:
    """A fairly basic optimiser designed to increase the value of scores for
    targeted property-based testing.

    This implements a fairly naive hill climbing algorithm based on randomly
    regenerating parts of the test case to attempt to improve the result. It is
    not expected to produce amazing results, because it is designed to be run
    in a fairly small testing budget, so it prioritises finding easy wins and
    bailing out quickly if that doesn't work.

    For more information about targeted property-based testing, see
    Löscher, Andreas, and Konstantinos Sagonas. "Targeted property-based
    testing." Proceedings of the 26th ACM SIGSOFT International Symposium on
    Software Testing and Analysis. ACM, 2017.
    """

    def __init__(self, engine, data, target, max_improvements=100):
        if False:
            for i in range(10):
                print('nop')
        'Optimise ``target`` starting from ``data``. Will stop either when\n        we seem to have found a local maximum or when the target score has\n        been improved ``max_improvements`` times. This limit is in place to\n        deal with the fact that the target score may not be bounded above.'
        self.engine = engine
        self.current_data = data
        self.target = target
        self.max_improvements = max_improvements
        self.improvements = 0

    def run(self):
        if False:
            print('Hello World!')
        self.hill_climb()

    def score_function(self, data):
        if False:
            i = 10
            return i + 15
        return data.target_observations.get(self.target, NO_SCORE)

    @property
    def current_score(self):
        if False:
            for i in range(10):
                print('nop')
        return self.score_function(self.current_data)

    def consider_new_test_data(self, data):
        if False:
            i = 10
            return i + 15
        'Consider a new data object as a candidate target. If it is better\n        than the current one, return True.'
        if data.status < Status.VALID:
            return False
        score = self.score_function(data)
        if score < self.current_score:
            return False
        if score > self.current_score:
            self.improvements += 1
            self.current_data = data
            return True
        assert score == self.current_score
        if len(data.buffer) <= len(self.current_data.buffer):
            self.current_data = data
            return True
        return False

    def hill_climb(self):
        if False:
            print('Hello World!')
        'The main hill climbing loop where we actually do the work: Take\n        data, and attempt to improve its score for target. select_example takes\n        a data object and returns an index to an example where we should focus\n        our efforts.'
        blocks_examined = set()
        prev = None
        i = len(self.current_data.blocks) - 1
        while i >= 0 and self.improvements <= self.max_improvements:
            if prev is not self.current_data:
                i = len(self.current_data.blocks) - 1
                prev = self.current_data
            if i in blocks_examined:
                i -= 1
                continue
            blocks_examined.add(i)
            data = self.current_data
            block = data.blocks[i]
            prefix = data.buffer[:block.start]
            existing = data.buffer[block.start:block.end]
            existing_as_int = int_from_bytes(existing)
            max_int_value = 256 ** len(existing) - 1
            if existing_as_int == max_int_value:
                continue

            def attempt_replace(v):
                if False:
                    for i in range(10):
                        print('nop')
                'Try replacing the current block in the current best test case\n                 with an integer of value i. Note that we use the *current*\n                best and not the one we started with. This helps ensure that\n                if we luck into a good draw when making random choices we get\n                to keep the good bits.'
                if v < 0 or v > max_int_value:
                    return False
                v_as_bytes = int_to_bytes(v, len(existing))
                for _ in range(3):
                    attempt = self.engine.cached_test_function(prefix + v_as_bytes + self.current_data.buffer[block.end:] + bytes(BUFFER_SIZE))
                    if self.consider_new_test_data(attempt):
                        return True
                    if attempt.status < Status.INVALID or len(attempt.buffer) == len(self.current_data.buffer):
                        return False
                    for (i, ex) in enumerate(self.current_data.examples):
                        if ex.start >= block.end:
                            break
                        if ex.end <= block.start:
                            continue
                        ex_attempt = attempt.examples[i]
                        if ex.length == ex_attempt.length:
                            continue
                        replacement = attempt.buffer[ex_attempt.start:ex_attempt.end]
                        if self.consider_new_test_data(self.engine.cached_test_function(prefix + replacement + self.current_data.buffer[ex.end:])):
                            return True
                return False
            if not attempt_replace(max_int_value):
                find_integer(lambda k: attempt_replace(k + existing_as_int))
            existing = self.current_data.buffer[block.start:block.end]
            existing_as_int = int_from_bytes(existing)
            if not attempt_replace(0):
                find_integer(lambda k: attempt_replace(existing_as_int - k))