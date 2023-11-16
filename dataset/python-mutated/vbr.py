from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Callable, Any, List, Tuple, Optional, Dict

class Actor:

    def __init__(self, uuid: str) -> None:
        if False:
            while True:
                i = 10
        self.uuid = uuid

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.uuid == other.uuid

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash(self.uuid)

class VerificationResult(IntEnum):
    SUCCESS = 0
    FAIL = 1
    UNDECIDED = 2

class NotAllowedError(Exception):
    pass

class MissingResultsError(Exception):
    pass

class UnknownActorError(Exception):
    pass

class AlreadyFinished(Exception):
    pass

class VerificationByRedundancy(ABC):

    def __init__(self, redundancy_factor: int, comparator: Callable[[Any, Any], bool], *_args, **_kwargs) -> None:
        if False:
            while True:
                i = 10
        self.redundancy_factor = redundancy_factor
        self.comparator = comparator

    @abstractmethod
    def add_actor(self, actor: Actor) -> None:
        if False:
            while True:
                i = 10
        'Caller informs class that this is the next actor he wants to assign\n        to the next subtask.\n        Raises:\n            NotAllowedError -- Actor given by caller is not allowed to compute\n            next task.\n            MissingResultsError -- Raised when caller wants to add next actor\n            but has already.\n            exhausted this method. Now the caller should provide results\n            by `add_result` method.\n        '
        pass

    @abstractmethod
    def add_result(self, actor: Actor, result: Optional[Any]) -> None:
        if False:
            return 10
        'Add a result for verification.\n        If a task computation has failed for some reason then the caller\n        should use this method with the result equal to None.\n        When user has added a result for each actor it reported by `add_actor`\n        a side effect might be the verdict being available or caller should\n        continue adding actors and results.\n        Arguments:\n            actor {Actor} -- Actor who has computed the result\n            result {Any} --  Computation result\n        Raises:\n            UnknownActorError - raised when caller deliver an actor that was\n            not previously reported by `add_actor` call.\n            ValueError - raised when attempting to add a result for some actor\n            more than once.\n        '
        pass

    @abstractmethod
    def get_verdicts(self) -> Optional[List[Tuple[Actor, Any, VerificationResult]]]:
        if False:
            while True:
                i = 10
        '\n        Returns:\n            Optional[List[Any, Actor, VerificationResult]] -- If verification\n            is resolved a list of 3-element tuples (actor, result reference,\n            verification_result) is returned. A None is returned when\n            verification has not been finished yet.\n        '
        pass

    @abstractmethod
    def validate_actor(self, actor):
        if False:
            while True:
                i = 10
        'Validates whether given actor is acceptable\n\n        Arguments:\n            actor {[type]} -- Actor to be validated\n        '
        pass

class Bucket:
    """A bucket containing a key and some values. Values are comparable
    directly, keys only by the comparator supplied at bucket creation"""

    def __init__(self, comparator: Callable[[Any, Any], bool], key: Any, value: Optional[Any]) -> None:
        if False:
            return 10
        self.comparator = comparator
        self.key = key
        if value is None:
            self.values: List[Any] = []
        else:
            self.values = [value]

    def key_equals(self, key: Any) -> bool:
        if False:
            print('Hello World!')
        return self.comparator(self.key, key)

    def try_add(self, key: Any, value: Any) -> bool:
        if False:
            print('Hello World!')
        'If the keys match, add value to the bucket and return True.\n        Otherwise return False'
        if self.key_equals(key):
            self.values.append(value)
            return True
        return False

    def __len__(self):
        if False:
            return 10
        return len(self.values)

class BucketVerifier(VerificationByRedundancy):

    def __init__(self, redundancy_factor: int, comparator: Callable[[Any, Any], bool], referee_count: int) -> None:
        if False:
            return 10
        super().__init__(redundancy_factor, comparator)
        self.actors: List[Actor] = []
        self.results: Dict[Actor, Any] = {}
        self.more_actors_needed = True
        self.buckets: List[Bucket] = []
        self.verdicts: Optional[List[Tuple[Actor, Any, VerificationResult]]] = None
        self.normal_actor_count = redundancy_factor + 1
        self.referee_count = referee_count
        self.majority = (self.normal_actor_count + self.referee_count) // 2 + 1
        self.max_actor_cnt = self.normal_actor_count + self.referee_count

    def validate_actor(self, actor):
        if False:
            i = 10
            return i + 15
        if actor in self.actors:
            raise NotAllowedError
        if not self.more_actors_needed:
            raise MissingResultsError

    def add_actor(self, actor):
        if False:
            for i in range(10):
                print('nop')
        self.validate_actor(actor)
        self.actors.append(actor)
        if len(self.actors) >= self.redundancy_factor + 1:
            self.more_actors_needed = False

    def remove_actor(self, actor):
        if False:
            i = 10
            return i + 15
        if self.verdicts is not None or actor in self.results.keys():
            raise AlreadyFinished
        self.actors.remove(actor)
        if len(self.actors) < self.redundancy_factor + 1:
            self.more_actors_needed = True

    def add_result(self, actor: Actor, result: Optional[Any]) -> None:
        if False:
            print('Hello World!')
        if actor not in self.actors:
            raise UnknownActorError
        if actor in self.results:
            raise ValueError
        self.results[actor] = result
        if result is not None:
            found = False
            for bucket in self.buckets:
                if bucket.try_add(key=result, value=actor):
                    found = True
                    break
            if not found:
                self.buckets.append(Bucket(self.comparator, key=result, value=actor))
        self.compute_verdicts()

    def get_verdicts(self) -> Optional[List[Tuple[Actor, Any, VerificationResult]]]:
        if False:
            i = 10
            return i + 15
        return self.verdicts

    def compute_verdicts(self) -> None:
        if False:
            i = 10
            return i + 15
        self.more_actors_needed = len(self.actors) < self.normal_actor_count
        if len(self.results) < self.normal_actor_count:
            self.verdicts = None
            return
        max_popularity = 0
        winners = None
        for bucket in self.buckets:
            max_popularity = max(max_popularity, len(bucket))
            if len(bucket) >= self.majority:
                winners = bucket.values
                break
        if winners:
            self.more_actors_needed = False
            success = VerificationResult.SUCCESS
            fail = VerificationResult.FAIL
            self.verdicts = [(actor, self.results[actor], success if actor in winners else fail) for actor in self.actors]
        elif self.majority - max_popularity <= self.referee_count and len(self.actors) < self.max_actor_cnt:
            self.verdicts = None
            self.more_actors_needed = True
        else:
            self.verdicts = [(actor, self.results[actor], VerificationResult.UNDECIDED) for actor in self.actors]
            self.more_actors_needed = False