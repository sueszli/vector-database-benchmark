"""Domain object for statistics models."""
from __future__ import annotations
import datetime
import json
import numbers
import sys
from core import feconf
from core import utils
from core.constants import constants
from core.domain import customization_args_util
from core.domain import exp_domain
from typing import Any, Dict, Final, List, Literal, Optional, TypedDict, Union
from core.domain import action_registry
from core.domain import interaction_registry
from core.domain import playthrough_issue_registry
MYPY = False
if MYPY:
    from core.domain import state_domain
MIGRATED_STATE_ANSWER_SESSION_ID_2017: Final = 'migrated_state_answer_session_id_2017'
MIGRATED_STATE_ANSWER_TIME_SPENT_IN_SEC: Final = 0.0
CALC_OUTPUT_TYPE_ANSWER_FREQUENCY_LIST: Final = 'AnswerFrequencyList'
CALC_OUTPUT_TYPE_CATEGORIZED_ANSWER_FREQUENCY_LISTS: Final = 'CategorizedAnswerFrequencyLists'
MAX_LEARNER_ANSWER_INFO_LIST_BYTE_SIZE: Final = 900000
MAX_ANSWER_DETAILS_BYTE_SIZE: Final = 10000
IssuesCustomizationArgsDictType = Dict[str, Dict[str, Union[str, int, List[str]]]]

class SubmittedAnswerDict(TypedDict):
    """Dictionary representing the SubmittedAnswer object."""
    answer: state_domain.AcceptableCorrectAnswerTypes
    time_spent_in_sec: float
    answer_group_index: int
    rule_spec_index: int
    classification_categorization: str
    session_id: str
    interaction_id: str
    params: Dict[str, Union[str, int]]
    rule_spec_str: Optional[str]
    answer_str: Optional[str]

class StateAnswersDict(TypedDict):
    """Dictionary representing the StateAnswers object."""
    exploration_id: str
    exploration_version: int
    state_name: str
    interaction_id: str
    submitted_answer_list: List[SubmittedAnswerDict]

class ExplorationIssueDict(TypedDict):
    """Dictionary representing the ExplorationIssue object."""
    issue_type: str
    issue_customization_args: IssuesCustomizationArgsDictType
    playthrough_ids: List[str]
    schema_version: int
    is_valid: bool

class PlaythroughDict(TypedDict):
    """Dictionary representing the PlayThrough object."""
    exp_id: str
    exp_version: int
    issue_type: str
    issue_customization_args: IssuesCustomizationArgsDictType
    actions: List[LearnerActionDict]

class ExplorationIssuesDict(TypedDict):
    """Dictionary representing the ExplorationIssues object."""
    exp_id: str
    exp_version: int
    unresolved_issues: List[ExplorationIssueDict]

class LearnerAnswerDetailsDict(TypedDict):
    """Dictionary representing the LearnerAnswerDetail object."""
    state_reference: str
    entity_type: str
    interaction_id: str
    learner_answer_info_list: List[LearnerAnswerInfoDict]
    accumulated_answer_info_json_size_bytes: int
    learner_answer_info_schema_version: int

class ExplorationStatsDict(TypedDict):
    """Dictionary representing the ExplorationStats object."""
    exp_id: str
    exp_version: int
    num_starts_v1: int
    num_starts_v2: int
    num_actual_starts_v1: int
    num_actual_starts_v2: int
    num_completions_v1: int
    num_completions_v2: int
    state_stats_mapping: Dict[str, Dict[str, int]]

class ExplorationStatsFrontendDict(TypedDict):
    """Dictionary representing the ExplorationStats object
    for use in frontend."""
    exp_id: str
    exp_version: int
    num_starts: int
    num_actual_starts: int
    num_completions: int
    state_stats_mapping: Dict[str, Dict[str, int]]

class LearnerActionDict(TypedDict):
    """Dictionary representing the LearnerAction object."""
    action_type: str
    action_customization_args: Dict[str, Dict[str, Union[str, int]]]
    schema_version: int

class AnswerOccurrenceDict(TypedDict):
    """Dictionary representing the AnswerOccurrence object."""
    answer: state_domain.AcceptableCorrectAnswerTypes
    frequency: int

class LearnerAnswerInfoDict(TypedDict):
    """Dictionary representing LearnerAnswerInfo object."""
    id: str
    answer: Optional[Union[str, int, Dict[str, str], List[str]]]
    answer_details: str
    created_on: str

class AggregatedStatsDict(TypedDict):
    """Dictionary representing aggregated_stats dict used to validate the
    SessionStateStats domain object."""
    num_starts: int
    num_actual_starts: int
    num_completions: int
    state_stats_mapping: Dict[str, Dict[str, int]]

class ExplorationStats:
    """Domain object representing analytics data for an exploration."""

    def __init__(self, exp_id: str, exp_version: int, num_starts_v1: int, num_starts_v2: int, num_actual_starts_v1: int, num_actual_starts_v2: int, num_completions_v1: int, num_completions_v2: int, state_stats_mapping: Dict[str, StateStats]) -> None:
        if False:
            print('Hello World!')
        'Constructs an ExplorationStats domain object.\n\n        Args:\n            exp_id: str. ID of the exploration.\n            exp_version: int. Version of the exploration.\n            num_starts_v1: int. Number of learners who started the exploration.\n            num_starts_v2: int. As above, but for events with version 2.\n            num_actual_starts_v1: int. Number of learners who actually attempted\n                the exploration. These are the learners who have completed the\n                initial state of the exploration and traversed to the next\n                state.\n            num_actual_starts_v2: int. As above, but for events with version 2.\n            num_completions_v1: int. Number of learners who completed the\n                exploration.\n            num_completions_v2: int. As above, but for events with version 2.\n            state_stats_mapping: dict. A dictionary mapping the state names of\n                an exploration to the corresponding StateStats domain object.\n        '
        self.exp_id = exp_id
        self.exp_version = exp_version
        self.num_starts_v1 = num_starts_v1
        self.num_starts_v2 = num_starts_v2
        self.num_actual_starts_v1 = num_actual_starts_v1
        self.num_actual_starts_v2 = num_actual_starts_v2
        self.num_completions_v1 = num_completions_v1
        self.num_completions_v2 = num_completions_v2
        self.state_stats_mapping = state_stats_mapping

    @property
    def num_starts(self) -> int:
        if False:
            print('Hello World!')
        'Returns the number of learners who started the exploration.\n\n        Returns:\n            int. The number of learners who started the exploration.\n        '
        return self.num_starts_v1 + self.num_starts_v2

    @property
    def num_actual_starts(self) -> int:
        if False:
            return 10
        'Returns the number of learners who actually attempted the\n        exploration. These are the learners who have completed the initial\n        state of the exploration and traversed to the next state.\n\n        Returns:\n            int. The number of learners who actually attempted the exploration.\n        '
        return self.num_actual_starts_v1 + self.num_actual_starts_v2

    @property
    def num_completions(self) -> int:
        if False:
            i = 10
            return i + 15
        'Returns the number of learners who completed the exploration.\n\n        Returns:\n            int. The number of learners who completed the exploration.\n        '
        return self.num_completions_v1 + self.num_completions_v2

    def to_dict(self) -> ExplorationStatsDict:
        if False:
            return 10
        'Returns a dict representation of the domain object.'
        state_stats_mapping_dict = {}
        for state_name in self.state_stats_mapping:
            state_stats_mapping_dict[state_name] = self.state_stats_mapping[state_name].to_dict()
        exploration_stats_dict: ExplorationStatsDict = {'exp_id': self.exp_id, 'exp_version': self.exp_version, 'num_starts_v1': self.num_starts_v1, 'num_starts_v2': self.num_starts_v2, 'num_actual_starts_v1': self.num_actual_starts_v1, 'num_actual_starts_v2': self.num_actual_starts_v2, 'num_completions_v1': self.num_completions_v1, 'num_completions_v2': self.num_completions_v2, 'state_stats_mapping': state_stats_mapping_dict}
        return exploration_stats_dict

    def to_frontend_dict(self) -> ExplorationStatsFrontendDict:
        if False:
            print('Hello World!')
        'Returns a dict representation of the domain object for use in the\n        frontend.\n        '
        state_stats_mapping_dict = {}
        for state_name in self.state_stats_mapping:
            state_stats_mapping_dict[state_name] = self.state_stats_mapping[state_name].to_frontend_dict()
        exploration_stats_dict: ExplorationStatsFrontendDict = {'exp_id': self.exp_id, 'exp_version': self.exp_version, 'num_starts': self.num_starts, 'num_actual_starts': self.num_actual_starts, 'num_completions': self.num_completions, 'state_stats_mapping': state_stats_mapping_dict}
        return exploration_stats_dict

    @classmethod
    def create_default(cls, exp_id: str, exp_version: int, state_stats_mapping: Dict[str, StateStats]) -> ExplorationStats:
        if False:
            for i in range(10):
                print('nop')
        'Creates a ExplorationStats domain object and sets all properties to\n        0.\n\n        Args:\n            exp_id: str. ID of the exploration.\n            exp_version: int. Version of the exploration.\n            state_stats_mapping: dict. A dict mapping state names to their\n                corresponding StateStats.\n\n        Returns:\n            ExplorationStats. The exploration stats domain object.\n        '
        return cls(exp_id, exp_version, 0, 0, 0, 0, 0, 0, state_stats_mapping)

    def get_sum_of_first_hit_counts(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Compute the sum of first hit counts for the exploration stats.\n\n        Returns:\n            int. Sum of first hit counts.\n        '
        sum_first_hits = 0
        for state_name in self.state_stats_mapping:
            state_stats = self.state_stats_mapping[state_name]
            sum_first_hits += state_stats.first_hit_count
        return sum_first_hits

    def validate(self) -> None:
        if False:
            print('Hello World!')
        'Validates the ExplorationStats domain object.'
        exploration_stats_properties: List[Literal['num_starts_v1', 'num_starts_v2', 'num_actual_starts_v1', 'num_actual_starts_v2', 'num_completions_v1', 'num_completions_v2']] = ['num_starts_v1', 'num_starts_v2', 'num_actual_starts_v1', 'num_actual_starts_v2', 'num_completions_v1', 'num_completions_v2']
        if not isinstance(self.exp_id, str):
            raise utils.ValidationError('Expected exp_id to be a string, received %s' % self.exp_id)
        if not isinstance(self.exp_version, int):
            raise utils.ValidationError('Expected exp_version to be an int, received %s' % self.exp_version)
        exploration_stats_dict = self.to_dict()
        for stat_property in exploration_stats_properties:
            if not isinstance(exploration_stats_dict[stat_property], int):
                raise utils.ValidationError('Expected %s to be an int, received %s' % (stat_property, exploration_stats_dict[stat_property]))
            if exploration_stats_dict[stat_property] < 0:
                raise utils.ValidationError('%s cannot have negative values' % stat_property)
        if not isinstance(self.state_stats_mapping, dict):
            raise utils.ValidationError('Expected state_stats_mapping to be a dict, received %s' % self.state_stats_mapping)

    def clone(self) -> ExplorationStats:
        if False:
            while True:
                i = 10
        'Returns a clone of this instance.'
        return ExplorationStats(self.exp_id, self.exp_version, self.num_starts_v1, self.num_starts_v2, self.num_actual_starts_v1, self.num_actual_starts_v2, self.num_completions_v1, self.num_completions_v2, {state_name: state_stats.clone() for (state_name, state_stats) in self.state_stats_mapping.items()})

class StateStats:
    """Domain object representing analytics data for an exploration's state.
    Instances of these domain objects pertain to the exploration ID and version
    as well.
    """

    def __init__(self, total_answers_count_v1: int, total_answers_count_v2: int, useful_feedback_count_v1: int, useful_feedback_count_v2: int, total_hit_count_v1: int, total_hit_count_v2: int, first_hit_count_v1: int, first_hit_count_v2: int, num_times_solution_viewed_v2: int, num_completions_v1: int, num_completions_v2: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Constructs a StateStats domain object.\n\n        Args:\n            total_answers_count_v1: int. Total number of answers submitted to\n                this state.\n            total_answers_count_v2: int. As above, but for events with version\n                2.\n            useful_feedback_count_v1: int. Total number of answers that received\n                useful feedback.\n            useful_feedback_count_v2: int. As above, but for events with version\n                2.\n            total_hit_count_v1: int. Total number of times the state was\n                entered.\n            total_hit_count_v2: int. As above, but for events with version 2.\n            first_hit_count_v1: int. Number of times the state was entered for\n                the first time.\n            first_hit_count_v2: int. As above, but for events with version 2.\n            num_times_solution_viewed_v2: int. Number of times the solution\n                button was triggered to answer a state (only for version 2).\n            num_completions_v1: int. Number of times the state was completed.\n            num_completions_v2: int. As above, but for events with version 2.\n        '
        self.total_answers_count_v1 = total_answers_count_v1
        self.total_answers_count_v2 = total_answers_count_v2
        self.useful_feedback_count_v1 = useful_feedback_count_v1
        self.useful_feedback_count_v2 = useful_feedback_count_v2
        self.total_hit_count_v1 = total_hit_count_v1
        self.total_hit_count_v2 = total_hit_count_v2
        self.first_hit_count_v1 = first_hit_count_v1
        self.first_hit_count_v2 = first_hit_count_v2
        self.num_times_solution_viewed_v2 = num_times_solution_viewed_v2
        self.num_completions_v1 = num_completions_v1
        self.num_completions_v2 = num_completions_v2

    @property
    def total_answers_count(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Returns the total number of answers submitted to this state.\n\n        Returns:\n            int. The total number of answers submitted to this state.\n        '
        return self.total_answers_count_v1 + self.total_answers_count_v2

    @property
    def useful_feedback_count(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Returns the total number of answers that received useful feedback.\n\n        Returns:\n            int. The total number of answers that received useful feedback.\n        '
        return self.useful_feedback_count_v1 + self.useful_feedback_count_v2

    @property
    def total_hit_count(self) -> int:
        if False:
            return 10
        'Returns the total number of times the state was entered.\n\n        Returns:\n            int. The total number of times the state was entered.\n        '
        return self.total_hit_count_v1 + self.total_hit_count_v2

    @property
    def first_hit_count(self) -> int:
        if False:
            return 10
        'Returns the number of times the state was entered for the first time.\n\n        Returns:\n            int. The number of times the state was entered for the first time.\n        '
        return self.first_hit_count_v1 + self.first_hit_count_v2

    @property
    def num_completions(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Returns total number of times the state was completed.\n\n        Returns:\n            int. The total number of times the state was completed.\n        '
        return self.num_completions_v1 + self.num_completions_v2

    @property
    def num_times_solution_viewed(self) -> int:
        if False:
            print('Hello World!')
        'Returns the number of times the solution button was triggered.\n\n        Returns:\n            int. Number of times the solution button was triggered to answer a\n            state only for events for schema version 2.\n        '
        return self.num_times_solution_viewed_v2

    @classmethod
    def create_default(cls) -> StateStats:
        if False:
            for i in range(10):
                print('nop')
        'Creates a StateStats domain object and sets all properties to 0.'
        return cls(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    def aggregate_from(self, other: Union[StateStats, SessionStateStats]) -> None:
        if False:
            i = 10
            return i + 15
        'Aggregates data from the other state stats into self.\n\n        Args:\n            other: StateStats | SessionStateStats. The other collection of stats\n                to aggregate from.\n\n        Raises:\n            TypeError. Given SessionStateStats can not be aggregated from.\n        '
        if isinstance(other, StateStats):
            self.total_answers_count_v1 += other.total_answers_count_v1
            self.total_answers_count_v2 += other.total_answers_count_v2
            self.useful_feedback_count_v1 += other.useful_feedback_count_v1
            self.useful_feedback_count_v2 += other.useful_feedback_count_v2
            self.total_hit_count_v1 += other.total_hit_count_v1
            self.total_hit_count_v2 += other.total_hit_count_v2
            self.first_hit_count_v1 += other.first_hit_count_v1
            self.first_hit_count_v2 += other.first_hit_count_v2
            self.num_times_solution_viewed_v2 += other.num_times_solution_viewed_v2
            self.num_completions_v1 += other.num_completions_v1
            self.num_completions_v2 += other.num_completions_v2
        elif isinstance(other, SessionStateStats):
            self.total_answers_count_v2 += other.total_answers_count
            self.useful_feedback_count_v2 += other.useful_feedback_count
            self.total_hit_count_v2 += other.total_hit_count
            self.first_hit_count_v2 += other.first_hit_count
            self.num_times_solution_viewed_v2 += other.num_times_solution_viewed
            self.num_completions_v2 += other.num_completions
        else:
            raise TypeError('%s can not be aggregated from' % (other.__class__.__name__,))

    def to_dict(self) -> Dict[str, int]:
        if False:
            return 10
        'Returns a dict representation of the domain object.'
        state_stats_dict = {'total_answers_count_v1': self.total_answers_count_v1, 'total_answers_count_v2': self.total_answers_count_v2, 'useful_feedback_count_v1': self.useful_feedback_count_v1, 'useful_feedback_count_v2': self.useful_feedback_count_v2, 'total_hit_count_v1': self.total_hit_count_v1, 'total_hit_count_v2': self.total_hit_count_v2, 'first_hit_count_v1': self.first_hit_count_v1, 'first_hit_count_v2': self.first_hit_count_v2, 'num_times_solution_viewed_v2': self.num_times_solution_viewed_v2, 'num_completions_v1': self.num_completions_v1, 'num_completions_v2': self.num_completions_v2}
        return state_stats_dict

    def to_frontend_dict(self) -> Dict[str, int]:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict representation of the domain object for use in the\n        frontend.\n        '
        state_stats_dict = {'total_answers_count': self.total_answers_count, 'useful_feedback_count': self.useful_feedback_count, 'total_hit_count': self.total_hit_count, 'first_hit_count': self.first_hit_count, 'num_times_solution_viewed': self.num_times_solution_viewed, 'num_completions': self.num_completions}
        return state_stats_dict

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        'Returns a detailed representation of self, distinguishing v1 values\n        from v2 values.\n\n        Returns:\n            str. A string representation of self.\n        '
        props = ['total_answers_count_v1', 'total_answers_count_v2', 'useful_feedback_count_v1', 'useful_feedback_count_v2', 'total_hit_count_v1', 'total_hit_count_v2', 'first_hit_count_v1', 'first_hit_count_v2', 'num_times_solution_viewed_v2', 'num_completions_v1', 'num_completions_v2']
        return '%s(%s)' % (self.__class__.__name__, ', '.join(('%s=%r' % (prop, getattr(self, prop)) for prop in props)))

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        'Returns a simple representation of self, combining v1 and v2 values.\n\n        Returns:\n            str. A string representation of self.\n        '
        props = ['total_answers_count', 'useful_feedback_count', 'total_hit_count', 'first_hit_count', 'num_times_solution_viewed', 'num_completions']
        return '%s(%s)' % (self.__class__.__name__, ', '.join(('%s=%r' % (prop, getattr(self, prop)) for prop in props)))

    def __eq__(self, other: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        'Implements == comparison between two StateStats instances, returning\n        whether they both hold the same values.\n\n        Args:\n            other: StateStats. The other instance to compare.\n\n        Returns:\n            bool. Whether the two instances have the same values.\n        '
        if not isinstance(other, StateStats):
            return NotImplemented
        return (self.total_answers_count_v1, self.total_answers_count_v2, self.useful_feedback_count_v1, self.useful_feedback_count_v2, self.total_hit_count_v1, self.total_hit_count_v2, self.first_hit_count_v1, self.first_hit_count_v2, self.num_times_solution_viewed_v2, self.num_completions_v1, self.num_completions_v2) == (other.total_answers_count_v1, other.total_answers_count_v2, other.useful_feedback_count_v1, other.useful_feedback_count_v2, other.total_hit_count_v1, other.total_hit_count_v2, other.first_hit_count_v1, other.first_hit_count_v2, other.num_times_solution_viewed_v2, other.num_completions_v1, other.num_completions_v2)

    def __hash__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Disallow hashing StateStats since they are mutable by design.'
        raise TypeError('%s is unhashable' % self.__class__.__name__)

    @classmethod
    def from_dict(cls, state_stats_dict: Dict[str, int]) -> StateStats:
        if False:
            while True:
                i = 10
        'Constructs a StateStats domain object from a dict.'
        return cls(state_stats_dict['total_answers_count_v1'], state_stats_dict['total_answers_count_v2'], state_stats_dict['useful_feedback_count_v1'], state_stats_dict['useful_feedback_count_v2'], state_stats_dict['total_hit_count_v1'], state_stats_dict['total_hit_count_v2'], state_stats_dict['first_hit_count_v1'], state_stats_dict['first_hit_count_v2'], state_stats_dict['num_times_solution_viewed_v2'], state_stats_dict['num_completions_v1'], state_stats_dict['num_completions_v2'])

    def validate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Validates the StateStats domain object.'
        state_stats_properties = ['total_answers_count_v1', 'total_answers_count_v2', 'useful_feedback_count_v1', 'useful_feedback_count_v2', 'total_hit_count_v1', 'total_hit_count_v2', 'first_hit_count_v1', 'first_hit_count_v2', 'num_times_solution_viewed_v2', 'num_completions_v1', 'num_completions_v2']
        state_stats_dict = self.to_dict()
        for stat_property in state_stats_properties:
            if not isinstance(state_stats_dict[stat_property], int):
                raise utils.ValidationError('Expected %s to be an int, received %s' % (stat_property, state_stats_dict[stat_property]))
            if state_stats_dict[stat_property] < 0:
                raise utils.ValidationError('%s cannot have negative values' % stat_property)

    def clone(self) -> StateStats:
        if False:
            return 10
        'Returns a clone of this instance.'
        return StateStats(self.total_answers_count_v1, self.total_answers_count_v2, self.useful_feedback_count_v1, self.useful_feedback_count_v2, self.total_hit_count_v1, self.total_hit_count_v2, self.first_hit_count_v1, self.first_hit_count_v2, self.num_times_solution_viewed_v2, self.num_completions_v1, self.num_completions_v2)

class SessionStateStats:
    """Domain object representing analytics data for a specific state of an
    exploration, aggregated during a continuous learner session.
    """

    def __init__(self, total_answers_count: int, useful_feedback_count: int, total_hit_count: int, first_hit_count: int, num_times_solution_viewed: int, num_completions: int):
        if False:
            print('Hello World!')
        'Constructs a SessionStateStats domain object.\n\n        Args:\n            total_answers_count: int. Total number of answers submitted to this\n                state.\n            useful_feedback_count: int. Total number of answers that received\n                useful feedback.\n            total_hit_count: int. Total number of times the state was entered.\n            first_hit_count: int. Number of times the state was entered for the\n                first time.\n            num_times_solution_viewed: int. Number of times the solution button\n                was triggered to answer a state.\n            num_completions: int. Number of times the state was completed.\n        '
        self.total_answers_count = total_answers_count
        self.useful_feedback_count = useful_feedback_count
        self.total_hit_count = total_hit_count
        self.first_hit_count = first_hit_count
        self.num_times_solution_viewed = num_times_solution_viewed
        self.num_completions = num_completions

    def __repr__(self) -> str:
        if False:
            return 10
        'Returns a detailed string representation of self.'
        props = ['total_answers_count', 'useful_feedback_count', 'total_hit_count', 'first_hit_count', 'num_times_solution_viewed', 'num_completions']
        return '%s(%s)' % (self.__class__.__name__, ', '.join(('%s=%r' % (prop, getattr(self, prop)) for prop in props)))

    def to_dict(self) -> Dict[str, int]:
        if False:
            print('Hello World!')
        'Returns a dict representation of self.'
        session_state_stats_dict = {'total_answers_count': self.total_answers_count, 'useful_feedback_count': self.useful_feedback_count, 'total_hit_count': self.total_hit_count, 'first_hit_count': self.first_hit_count, 'num_times_solution_viewed': self.num_times_solution_viewed, 'num_completions': self.num_completions}
        return session_state_stats_dict

    @staticmethod
    def validate_aggregated_stats_dict(aggregated_stats: AggregatedStatsDict) -> AggregatedStatsDict:
        if False:
            print('Hello World!')
        'Validates the SessionStateStats domain object.\n\n        Args:\n            aggregated_stats: dict. The aggregated stats dict to validate.\n\n        Returns:\n            aggregated_stats: dict. The validated aggregated stats dict.\n\n        Raises:\n            ValidationError. Whether the aggregated_stats dict is invalid.\n        '
        exploration_stats_properties = ['num_starts', 'num_actual_starts', 'num_completions']
        state_stats_properties = ['total_answers_count', 'useful_feedback_count', 'total_hit_count', 'first_hit_count', 'num_times_solution_viewed', 'num_completions']
        for exp_stats_property in exploration_stats_properties:
            if exp_stats_property not in aggregated_stats:
                raise utils.ValidationError('%s not in aggregated stats dict.' % exp_stats_property)
            if not isinstance(aggregated_stats[exp_stats_property], int):
                raise utils.ValidationError('Expected %s to be an int, received %s' % (exp_stats_property, aggregated_stats[exp_stats_property]))
        state_stats_mapping = aggregated_stats['state_stats_mapping']
        for state_name in state_stats_mapping:
            for state_stats_property in state_stats_properties:
                if state_stats_property not in state_stats_mapping[state_name]:
                    raise utils.ValidationError('%s not in state stats mapping of %s in aggregated stats dict.' % (state_stats_property, state_name))
                if not isinstance(state_stats_mapping[state_name][state_stats_property], int):
                    state_stats = state_stats_mapping[state_name]
                    raise utils.ValidationError('Expected %s to be an int, received %s' % (state_stats_property, state_stats[state_stats_property]))
        return aggregated_stats

    def __eq__(self, other: Any) -> Any:
        if False:
            i = 10
            return i + 15
        'Implements == comparison between two SessionStateStats instances,\n        returning whether they hold the same values.\n\n        Args:\n            other: SessionStateStats. The other instance to compare.\n\n        Returns:\n            bool. Whether the two instances have the same values.\n        '
        if not isinstance(other, SessionStateStats):
            return NotImplemented
        return (self.total_answers_count, self.useful_feedback_count, self.total_hit_count, self.first_hit_count, self.num_times_solution_viewed, self.num_completions) == (other.total_answers_count, other.useful_feedback_count, other.total_hit_count, other.first_hit_count, other.num_times_solution_viewed, other.num_completions)

    def __hash__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Disallow hashing SessionStateStats since it is mutable by design.'
        raise TypeError('%s is unhashable' % self.__class__.__name__)

    @classmethod
    def create_default(cls) -> SessionStateStats:
        if False:
            return 10
        'Creates a SessionStateStats domain object with all values at 0.'
        return cls(0, 0, 0, 0, 0, 0)

    @classmethod
    def from_dict(cls, session_state_stats_dict: Dict[str, int]) -> SessionStateStats:
        if False:
            return 10
        'Creates a SessionStateStats domain object from the given dict.'
        return cls(session_state_stats_dict['total_answers_count'], session_state_stats_dict['useful_feedback_count'], session_state_stats_dict['total_hit_count'], session_state_stats_dict['first_hit_count'], session_state_stats_dict['num_times_solution_viewed'], session_state_stats_dict['num_completions'])

class ExplorationIssues:
    """Domain object representing the exploration to issues mapping for an
    exploration.
    """

    def __init__(self, exp_id: str, exp_version: int, unresolved_issues: List[ExplorationIssue]) -> None:
        if False:
            while True:
                i = 10
        'Constructs an ExplorationIssues domain object.\n\n        Args:\n            exp_id: str. ID of the exploration.\n            exp_version: int. Version of the exploration.\n            unresolved_issues: list(ExplorationIssue). List of exploration\n                issues.\n        '
        self.exp_id = exp_id
        self.exp_version = exp_version
        self.unresolved_issues = unresolved_issues

    @classmethod
    def create_default(cls, exp_id: str, exp_version: int) -> ExplorationIssues:
        if False:
            print('Hello World!')
        'Creates a default ExplorationIssues domain object.\n\n        Args:\n            exp_id: str. ID of the exploration.\n            exp_version: int. Version of the exploration.\n\n        Returns:\n            ExplorationIssues. The exploration issues domain object.\n        '
        return cls(exp_id, exp_version, [])

    def to_dict(self) -> ExplorationIssuesDict:
        if False:
            return 10
        'Returns a dict representation of the ExplorationIssues domain object.\n\n        Returns:\n            dict. A dict mapping of all fields of ExplorationIssues object.\n        '
        unresolved_issue_dicts = [unresolved_issue.to_dict() for unresolved_issue in self.unresolved_issues]
        return {'exp_id': self.exp_id, 'exp_version': self.exp_version, 'unresolved_issues': unresolved_issue_dicts}

    @classmethod
    def from_dict(cls, exp_issues_dict: ExplorationIssuesDict) -> ExplorationIssues:
        if False:
            print('Hello World!')
        'Returns an ExplorationIssues object from a dict.\n\n        Args:\n            exp_issues_dict: dict. A dict mapping of all fields of\n                ExplorationIssues object.\n\n        Returns:\n            ExplorationIssues. The corresponding ExplorationIssues domain\n            object.\n        '
        unresolved_issues = [ExplorationIssue.from_dict(unresolved_issue_dict) for unresolved_issue_dict in exp_issues_dict['unresolved_issues']]
        return cls(exp_issues_dict['exp_id'], exp_issues_dict['exp_version'], unresolved_issues)

    def validate(self) -> None:
        if False:
            while True:
                i = 10
        'Validates the ExplorationIssues domain object.'
        if not isinstance(self.exp_id, str):
            raise utils.ValidationError('Expected exp_id to be a string, received %s' % type(self.exp_id))
        if not isinstance(self.exp_version, int):
            raise utils.ValidationError('Expected exp_version to be an int, received %s' % type(self.exp_version))
        if not isinstance(self.unresolved_issues, list):
            raise utils.ValidationError('Expected unresolved_issues to be a list, received %s' % type(self.unresolved_issues))
        for issue in self.unresolved_issues:
            issue.validate()

class Playthrough:
    """Domain object representing a learner playthrough."""

    def __init__(self, exp_id: str, exp_version: int, issue_type: str, issue_customization_args: IssuesCustomizationArgsDictType, actions: List[LearnerAction]):
        if False:
            i = 10
            return i + 15
        'Constructs a Playthrough domain object.\n\n        Args:\n            exp_id: str. ID of the exploration.\n            exp_version: int. Version of the exploration.\n            issue_type: str. Type of the issue.\n            issue_customization_args: dict. The customization args dict for the\n                given issue_type.\n            actions: list(LearnerAction). List of playthrough learner actions.\n        '
        self.exp_id = exp_id
        self.exp_version = exp_version
        self.issue_type = issue_type
        self.issue_customization_args = issue_customization_args
        self.actions = actions

    def to_dict(self) -> PlaythroughDict:
        if False:
            return 10
        'Returns a dict representation of the Playthrough domain object.\n\n        Returns:\n            dict. A dict mapping of all fields of Playthrough object.\n        '
        action_dicts = [action.to_dict() for action in self.actions]
        return {'exp_id': self.exp_id, 'exp_version': self.exp_version, 'issue_type': self.issue_type, 'issue_customization_args': self.issue_customization_args, 'actions': action_dicts}

    @classmethod
    def from_dict(cls, playthrough_data: PlaythroughDict) -> Playthrough:
        if False:
            i = 10
            return i + 15
        'Checks whether the playthrough dict has the correct keys and then\n        returns a domain object instance.\n\n        Args:\n            playthrough_data: dict. A dict mapping of all fields of Playthrough\n                object.\n\n        Returns:\n            Playthrough. The corresponding Playthrough domain object.\n        '
        playthrough_properties = ['exp_id', 'exp_version', 'issue_type', 'issue_customization_args', 'actions']
        for playthrough_property in playthrough_properties:
            if playthrough_property not in playthrough_data:
                raise utils.ValidationError('%s not in playthrough data dict.' % playthrough_property)
        actions = [LearnerAction.from_dict(action_dict) for action_dict in playthrough_data['actions']]
        playthrough = cls(playthrough_data['exp_id'], playthrough_data['exp_version'], playthrough_data['issue_type'], playthrough_data['issue_customization_args'], actions)
        playthrough.validate()
        return playthrough

    def validate(self) -> None:
        if False:
            return 10
        'Validates the Playthrough domain object.'
        if not isinstance(self.exp_id, str):
            raise utils.ValidationError('Expected exp_id to be a string, received %s' % type(self.exp_id))
        if not isinstance(self.exp_version, int):
            raise utils.ValidationError('Expected exp_version to be an int, received %s' % type(self.exp_version))
        if not isinstance(self.issue_type, str):
            raise utils.ValidationError('Expected issue_type to be a string, received %s' % type(self.issue_type))
        if not isinstance(self.issue_customization_args, dict):
            raise utils.ValidationError('Expected issue_customization_args to be a dict, received %s' % type(self.issue_customization_args))
        try:
            issue = playthrough_issue_registry.Registry.get_issue_by_type(self.issue_type)
        except KeyError as e:
            raise utils.ValidationError('Invalid issue type: %s' % self.issue_type) from e
        customization_args_util.validate_customization_args_and_values('issue', self.issue_type, self.issue_customization_args, issue.customization_arg_specs)
        if not isinstance(self.actions, list):
            raise utils.ValidationError('Expected actions to be a list, received %s' % type(self.actions))
        for action in self.actions:
            action.validate()

class ExplorationIssue:
    """Domain object representing an exploration issue."""

    def __init__(self, issue_type: str, issue_customization_args: IssuesCustomizationArgsDictType, playthrough_ids: List[str], schema_version: int, is_valid: bool):
        if False:
            while True:
                i = 10
        "Constructs an ExplorationIssue domain object.\n\n        Args:\n            issue_type: str. Type of the issue.\n            issue_customization_args: dict. The customization dict. The keys are\n                names of customization_args and the values are dicts with a\n                single key, 'value', whose corresponding value is the value of\n                the customization arg.\n            playthrough_ids: list(str). List of playthrough IDs.\n            schema_version: int. Schema version for the exploration issue.\n            is_valid: bool. Whether the issue and the associated playthroughs\n                are valid.\n        "
        self.issue_type = issue_type
        self.issue_customization_args = issue_customization_args
        self.playthrough_ids = playthrough_ids
        self.schema_version = schema_version
        self.is_valid = is_valid

    def __eq__(self, other: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, ExplorationIssue):
            return NotImplemented
        return self.issue_type == other.issue_type and self.issue_customization_args == other.issue_customization_args and (self.playthrough_ids == other.playthrough_ids) and (self.schema_version == other.schema_version) and (self.is_valid == other.is_valid)

    def to_dict(self) -> ExplorationIssueDict:
        if False:
            print('Hello World!')
        'Returns a dict representation of the ExplorationIssue domain object.\n\n        Returns:\n            dict. A dict mapping of all fields of ExplorationIssue object.\n        '
        return {'issue_type': self.issue_type, 'issue_customization_args': self.issue_customization_args, 'playthrough_ids': self.playthrough_ids, 'schema_version': self.schema_version, 'is_valid': self.is_valid}

    @classmethod
    def from_dict(cls, exp_issue_dict: ExplorationIssueDict) -> ExplorationIssue:
        if False:
            i = 10
            return i + 15
        'Checks whether the exploration issue dict has the correct keys and\n        then returns a domain object instance.\n\n        Args:\n            exp_issue_dict: dict. A dict mapping of all fields of\n                ExplorationIssue object.\n\n        Returns:\n            ExplorationIssue. The corresponding ExplorationIssue domain object.\n        '
        exp_issue_properties = ['issue_type', 'schema_version', 'issue_customization_args', 'playthrough_ids', 'is_valid']
        for exp_issue_property in exp_issue_properties:
            if exp_issue_property not in exp_issue_dict:
                raise utils.ValidationError('%s not in exploration issue dict.' % exp_issue_property)
        exp_issue = cls(exp_issue_dict['issue_type'], exp_issue_dict['issue_customization_args'], exp_issue_dict['playthrough_ids'], exp_issue_dict['schema_version'], exp_issue_dict['is_valid'])
        exp_issue.validate()
        return exp_issue

    @classmethod
    def update_exp_issue_from_model(cls, issue_dict: ExplorationIssueDict) -> None:
        if False:
            return 10
        'Converts the exploration issue blob given from\n        current issue_schema_version to current issue_schema_version + 1.\n        Note that the issue_dict being passed in is modified in-place.\n\n        Args:\n            issue_dict: dict. Dict representing the ExplorationIssue object.\n        '
        current_issue_schema_version = issue_dict['schema_version']
        issue_dict['schema_version'] += 1
        conversion_fn = getattr(cls, '_convert_issue_v%s_dict_to_v%s_dict' % (current_issue_schema_version, current_issue_schema_version + 1))
        issue_dict = conversion_fn(issue_dict)

    @classmethod
    def _convert_issue_v1_dict_to_v2_dict(cls, issue_dict: Dict[str, Union[str, Dict[str, Dict[str, str]], List[str], int, bool]]) -> None:
        if False:
            return 10
        'Converts a v1 issue dict to a v2 issue dict. This function is now\n        implemented only for testing purposes and must be rewritten when an\n        actual schema migration from v1 to v2 takes place.\n        '
        raise NotImplementedError('The _convert_issue_v1_dict_to_v2_dict() method is missing from the derived class. It should be implemented in the derived class.')

    def validate(self) -> None:
        if False:
            print('Hello World!')
        'Validates the ExplorationIssue domain object.'
        if not isinstance(self.issue_type, str):
            raise utils.ValidationError('Expected issue_type to be a string, received %s' % type(self.issue_type))
        if not isinstance(self.schema_version, int):
            raise utils.ValidationError('Expected schema_version to be an int, received %s' % type(self.schema_version))
        try:
            issue = playthrough_issue_registry.Registry.get_issue_by_type(self.issue_type)
        except KeyError as e:
            raise utils.ValidationError('Invalid issue type: %s' % self.issue_type) from e
        customization_args_util.validate_customization_args_and_values('issue', self.issue_type, self.issue_customization_args, issue.customization_arg_specs)
        if not isinstance(self.playthrough_ids, list):
            raise utils.ValidationError('Expected playthrough_ids to be a list, received %s' % type(self.playthrough_ids))
        for playthrough_id in self.playthrough_ids:
            if not isinstance(playthrough_id, str):
                raise utils.ValidationError('Expected each playthrough_id to be a string, received %s' % type(playthrough_id))

class LearnerAction:
    """Domain object representing a learner action."""

    def __init__(self, action_type: str, action_customization_args: Dict[str, Dict[str, Union[str, int]]], schema_version: int):
        if False:
            for i in range(10):
                print('nop')
        "Constructs a LearnerAction domain object.\n\n        Args:\n            action_type: str. Type of the action.\n            action_customization_args: dict. The customization dict. The keys\n                are names of customization_args and the values are dicts with a\n                single key, 'value', whose corresponding value is the value of\n                the customization arg.\n            schema_version: int. Schema version for the learner action.\n        "
        self.action_type = action_type
        self.action_customization_args = action_customization_args
        self.schema_version = schema_version

    def to_dict(self) -> LearnerActionDict:
        if False:
            while True:
                i = 10
        'Returns a dict representation of the LearnerAction domain object.\n\n        Returns:\n            dict. A dict mapping of all fields of LearnerAction object.\n        '
        return {'action_type': self.action_type, 'action_customization_args': self.action_customization_args, 'schema_version': self.schema_version}

    @classmethod
    def from_dict(cls, action_dict: LearnerActionDict) -> LearnerAction:
        if False:
            while True:
                i = 10
        'Returns a LearnerAction object from a dict.\n\n        Args:\n            action_dict: dict. A dict mapping of all fields of LearnerAction\n                object.\n\n        Returns:\n            LearnerAction. The corresponding LearnerAction domain object.\n        '
        return cls(action_dict['action_type'], action_dict['action_customization_args'], action_dict['schema_version'])

    @classmethod
    def update_learner_action_from_model(cls, action_dict: LearnerActionDict) -> None:
        if False:
            return 10
        'Converts the learner action blob given from\n        current action_schema_version to current action_schema_version + 1.\n        Note that the action_dict being passed in is modified in-place.\n\n        Args:\n            action_dict: dict. Dict representing the LearnerAction object.\n        '
        current_action_schema_version = action_dict['schema_version']
        action_dict['schema_version'] += 1
        conversion_fn = getattr(cls, '_convert_action_v%s_dict_to_v%s_dict' % (current_action_schema_version, current_action_schema_version + 1))
        action_dict = conversion_fn(action_dict)

    @classmethod
    def _convert_action_v1_dict_to_v2_dict(cls, action_dict: LearnerActionDict) -> None:
        if False:
            i = 10
            return i + 15
        'Converts a v1 action dict to a v2 action dict. This function is now\n        implemented only for testing purposes and must be rewritten when an\n        actual schema migration from v1 to v2 takes place.\n        '
        raise NotImplementedError('The _convert_action_v1_dict_to_v2_dict() method is missing from the derived class. It should be implemented in the derived class.')

    def validate(self) -> None:
        if False:
            i = 10
            return i + 15
        'Validates the LearnerAction domain object.'
        if not isinstance(self.action_type, str):
            raise utils.ValidationError('Expected action_type to be a string, received %s' % type(self.action_type))
        if not isinstance(self.schema_version, int):
            raise utils.ValidationError('Expected schema_version to be an int, received %s' % type(self.schema_version))
        try:
            action = action_registry.Registry.get_action_by_type(self.action_type)
        except KeyError as e:
            raise utils.ValidationError('Invalid action type: %s' % self.action_type) from e
        customization_args_util.validate_customization_args_and_values('action', self.action_type, self.action_customization_args, action.customization_arg_specs)

class StateAnswers:
    """Domain object containing answers submitted to an exploration state."""

    def __init__(self, exploration_id: str, exploration_version: int, state_name: str, interaction_id: str, submitted_answer_list: List[SubmittedAnswer], schema_version: int=feconf.CURRENT_STATE_ANSWERS_SCHEMA_VERSION) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Constructs a StateAnswers domain object.\n\n        Args:\n            exploration_id: str. The ID of the exploration corresponding to\n                submitted answers.\n            exploration_version: int. The version of the exploration\n                corresponding to submitted answers.\n            state_name: str. The state to which the answers were submitted.\n            interaction_id: str. The ID of the interaction which created the\n                answers.\n            submitted_answer_list: list. The list of SubmittedAnswer domain\n                objects that were submitted to the exploration and version\n                specified in this object.\n            schema_version: int. The schema version of this answers object.\n        '
        self.exploration_id = exploration_id
        self.exploration_version = exploration_version
        self.state_name = state_name
        self.interaction_id = interaction_id
        self.submitted_answer_list = submitted_answer_list
        self.schema_version = schema_version

    def get_submitted_answer_dict_list(self) -> List[SubmittedAnswerDict]:
        if False:
            while True:
                i = 10
        'Returns the submitted_answer_list stored within this object as a list\n        of StateAnswer dicts.\n        '
        return [state_answer.to_dict() for state_answer in self.submitted_answer_list]

    def validate(self) -> None:
        if False:
            i = 10
            return i + 15
        'Validates StateAnswers domain object entity.'
        if not isinstance(self.exploration_id, str):
            raise utils.ValidationError('Expected exploration_id to be a string, received %s' % str(self.exploration_id))
        if not isinstance(self.state_name, str):
            raise utils.ValidationError('Expected state_name to be a string, received %s' % str(self.state_name))
        if self.interaction_id is not None:
            if not isinstance(self.interaction_id, str):
                raise utils.ValidationError('Expected interaction_id to be a string, received %s' % str(self.interaction_id))
            if self.interaction_id not in interaction_registry.Registry.get_all_interaction_ids():
                raise utils.ValidationError('Unknown interaction_id: %s' % self.interaction_id)
        if not isinstance(self.submitted_answer_list, list):
            raise utils.ValidationError('Expected submitted_answer_list to be a list, received %s' % str(self.submitted_answer_list))
        if not isinstance(self.schema_version, int):
            raise utils.ValidationError('Expected schema_version to be an integer, received %s' % str(self.schema_version))
        if self.schema_version < 1:
            raise utils.ValidationError('schema_version < 1: %d' % self.schema_version)
        if self.schema_version > feconf.CURRENT_STATE_ANSWERS_SCHEMA_VERSION:
            raise utils.ValidationError('schema_version > feconf.CURRENT_STATE_ANSWERS_SCHEMA_VERSION (%d): %d' % (feconf.CURRENT_STATE_ANSWERS_SCHEMA_VERSION, self.schema_version))

class SubmittedAnswer:
    """Domain object representing an answer submitted to a state."""

    def __init__(self, answer: state_domain.AcceptableCorrectAnswerTypes, interaction_id: str, answer_group_index: int, rule_spec_index: int, classification_categorization: str, params: Dict[str, Union[str, int]], session_id: str, time_spent_in_sec: float, rule_spec_str: Optional[str]=None, answer_str: Optional[str]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.answer = answer
        self.interaction_id = interaction_id
        self.answer_group_index = answer_group_index
        self.rule_spec_index = rule_spec_index
        self.classification_categorization = classification_categorization
        self.params = params
        self.session_id = session_id
        self.time_spent_in_sec = time_spent_in_sec
        self.rule_spec_str = rule_spec_str
        self.answer_str = answer_str

    def to_dict(self) -> SubmittedAnswerDict:
        if False:
            return 10
        'Returns the dict of submitted answer.\n\n        Returns:\n            dict. The submitted answer dict.\n        '
        submitted_answer_dict: SubmittedAnswerDict = {'answer': self.answer, 'interaction_id': self.interaction_id, 'answer_group_index': self.answer_group_index, 'rule_spec_index': self.rule_spec_index, 'classification_categorization': self.classification_categorization, 'params': self.params, 'session_id': self.session_id, 'time_spent_in_sec': self.time_spent_in_sec, 'rule_spec_str': self.rule_spec_str, 'answer_str': self.answer_str}
        if self.rule_spec_str is not None:
            submitted_answer_dict['rule_spec_str'] = self.rule_spec_str
        if self.answer_str is not None:
            submitted_answer_dict['answer_str'] = self.answer_str
        return submitted_answer_dict

    @classmethod
    def from_dict(cls, submitted_answer_dict: SubmittedAnswerDict) -> SubmittedAnswer:
        if False:
            for i in range(10):
                print('nop')
        'Returns the domain object representing an answer submitted to a\n        state.\n\n        Returns:\n            SubmittedAnswer. The SubmittedAnswer domin object.\n        '
        return cls(submitted_answer_dict['answer'], submitted_answer_dict['interaction_id'], submitted_answer_dict['answer_group_index'], submitted_answer_dict['rule_spec_index'], submitted_answer_dict['classification_categorization'], submitted_answer_dict['params'], submitted_answer_dict['session_id'], submitted_answer_dict['time_spent_in_sec'], rule_spec_str=submitted_answer_dict.get('rule_spec_str'), answer_str=submitted_answer_dict.get('answer_str'))

    def validate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Validates this submitted answer object.'
        if self.time_spent_in_sec is None:
            raise utils.ValidationError('SubmittedAnswers must have a provided time_spent_in_sec')
        if self.session_id is None:
            raise utils.ValidationError('SubmittedAnswers must have a provided session_id')
        if self.rule_spec_str is not None and (not isinstance(self.rule_spec_str, str)):
            raise utils.ValidationError('Expected rule_spec_str to be either None or a string, received %s' % str(self.rule_spec_str))
        if self.answer_str is not None and (not isinstance(self.answer_str, str)):
            raise utils.ValidationError('Expected answer_str to be either None or a string, received %s' % str(self.answer_str))
        if not isinstance(self.session_id, str):
            raise utils.ValidationError('Expected session_id to be a string, received %s' % str(self.session_id))
        if not isinstance(self.time_spent_in_sec, numbers.Number):
            raise utils.ValidationError('Expected time_spent_in_sec to be a number, received %s' % str(self.time_spent_in_sec))
        if not isinstance(self.params, dict):
            raise utils.ValidationError('Expected params to be a dict, received %s' % str(self.params))
        if not isinstance(self.answer_group_index, int):
            raise utils.ValidationError('Expected answer_group_index to be an integer, received %s' % str(self.answer_group_index))
        if self.rule_spec_index is not None and (not isinstance(self.rule_spec_index, int)):
            raise utils.ValidationError('Expected rule_spec_index to be an integer, received %s' % str(self.rule_spec_index))
        if self.answer_group_index < 0:
            raise utils.ValidationError('Expected answer_group_index to be non-negative, received %d' % self.answer_group_index)
        if self.rule_spec_index is not None and self.rule_spec_index < 0:
            raise utils.ValidationError('Expected rule_spec_index to be non-negative, received %d' % self.rule_spec_index)
        if self.time_spent_in_sec < 0.0:
            raise utils.ValidationError('Expected time_spent_in_sec to be non-negative, received %f' % self.time_spent_in_sec)
        if self.answer is None and self.interaction_id not in feconf.LINEAR_INTERACTION_IDS:
            raise utils.ValidationError('SubmittedAnswers must have a provided answer except for linear interactions')
        valid_classification_categories = [exp_domain.EXPLICIT_CLASSIFICATION, exp_domain.TRAINING_DATA_CLASSIFICATION, exp_domain.STATISTICAL_CLASSIFICATION, exp_domain.DEFAULT_OUTCOME_CLASSIFICATION]
        if self.classification_categorization not in valid_classification_categories:
            raise utils.ValidationError('Expected valid classification_categorization, received %s' % self.classification_categorization)

class AnswerOccurrence:
    """Domain object that represents a specific answer that occurred some number
    of times.
    """

    def __init__(self, answer: state_domain.AcceptableCorrectAnswerTypes, frequency: int) -> None:
        if False:
            while True:
                i = 10
        'Initialize domain object for answer occurrences.'
        self.answer = answer
        self.frequency = frequency

    def to_raw_type(self) -> AnswerOccurrenceDict:
        if False:
            print('Hello World!')
        "Returns a Python dict representing the specific answer.\n\n        Returns:\n            dict. The specific answer dict in the following format:\n            {\n                'answer': *. The answer submitted by the learner.\n                'frequency': int. The number of occurrences of the answer.\n            }\n        "
        return {'answer': self.answer, 'frequency': self.frequency}

    @classmethod
    def from_raw_type(cls, answer_occurrence_dict: AnswerOccurrenceDict) -> AnswerOccurrence:
        if False:
            for i in range(10):
                print('nop')
        "Returns domain object that represents a specific answer that occurred\n        some number of times.\n\n        Args:\n            answer_occurrence_dict: dict. The specific answer dict in the\n                following format:\n                {\n                    'answer': *. The answer submitted by the learner.\n                    'frequency': int. The number of occurrences of the answer.\n                }\n\n        Returns:\n            AnswerOccurrence. The AnswerOccurrence domain object.\n        "
        return cls(answer_occurrence_dict['answer'], answer_occurrence_dict['frequency'])

class AnswerCalculationOutput:
    """Domain object superclass that represents the output of an answer
    calculation.
    """

    def __init__(self, calculation_output_type: str):
        if False:
            for i in range(10):
                print('nop')
        self.calculation_output_type = calculation_output_type

class AnswerFrequencyList(AnswerCalculationOutput):
    """Domain object that represents an output list of AnswerOccurrences."""

    def __init__(self, answer_occurrences: Optional[List[AnswerOccurrence]]=None) -> None:
        if False:
            while True:
                i = 10
        'Initialize domain object for answer frequency list for a given list\n        of AnswerOccurrence objects (default is empty list).\n        '
        super().__init__(CALC_OUTPUT_TYPE_ANSWER_FREQUENCY_LIST)
        self.answer_occurrences = answer_occurrences if answer_occurrences else []

    def to_raw_type(self) -> List[AnswerOccurrenceDict]:
        if False:
            return 10
        "Returns the answer occurrences list with each answer represented as\n        a Python dict.\n\n        Returns:\n            list(dict). A list of answer occurrence dicts. Each dict has the\n            following format:\n            {\n                'answer': *. The answer submitted by the learner.\n                'frequency': int. The number of occurrences of the answer.\n            }\n        "
        return [answer_occurrence.to_raw_type() for answer_occurrence in self.answer_occurrences]

    @classmethod
    def from_raw_type(cls, answer_occurrence_list: List[AnswerOccurrenceDict]) -> AnswerFrequencyList:
        if False:
            i = 10
            return i + 15
        "Creates a domain object that represents an output list of\n        AnswerOccurrences.\n\n        Args:\n            answer_occurrence_list: list(dict). A list containing answer\n                occurrence dicts in the following format:\n                {\n                    'answer': *. The answer submitted by the learner.\n                    'frequency': int. The number of occurrences of the answer.\n                }\n\n        Returns:\n            AnswerFrequencyList. The domain object for answer occurrences list.\n        "
        return cls([AnswerOccurrence.from_raw_type(answer_occurrence_dict) for answer_occurrence_dict in answer_occurrence_list])

class CategorizedAnswerFrequencyLists(AnswerCalculationOutput):
    """AnswerFrequencyLists that are categorized based on arbitrary
    categories.
    """

    def __init__(self, categorized_answer_freq_lists: Optional[Dict[str, AnswerFrequencyList]]=None) -> None:
        if False:
            i = 10
            return i + 15
        'Initialize domain object for categorized answer frequency lists for\n        a given dict (default is empty).\n        '
        super().__init__(CALC_OUTPUT_TYPE_CATEGORIZED_ANSWER_FREQUENCY_LISTS)
        self.categorized_answer_freq_lists = categorized_answer_freq_lists if categorized_answer_freq_lists else {}

    def to_raw_type(self) -> Dict[str, List[AnswerOccurrenceDict]]:
        if False:
            return 10
        "Returns the categorized frequency Python dict.\n\n        Returns:\n            dict. A dict whose keys are category names and whose corresponding\n            values are lists of answer frequency dicts. Each answer\n            frequency dict has the following keys and values:\n            {\n                'answer': *. The answer submitted by the learner.\n                'frequency': int. The number of occurrences of the answer.\n            }\n        "
        return {category: answer_frequency_list.to_raw_type() for (category, answer_frequency_list) in self.categorized_answer_freq_lists.items()}

    @classmethod
    def from_raw_type(cls, categorized_frequency_dict: Dict[str, List[AnswerOccurrenceDict]]) -> CategorizedAnswerFrequencyLists:
        if False:
            i = 10
            return i + 15
        "Returns the domain object for categorized answer frequency dict for\n        a given dict.\n\n        Args:\n            categorized_frequency_dict: dict. The categorized answer frequency\n                dict whose keys are category names and whose corresponding\n                values are lists of answer frequency dicts. Each answer\n                frequency dict has the following keys and values:\n                {\n                    'answer': *. The answer submitted by the learner.\n                    'frequency': int. The number of occurrences of the answer.\n                }\n\n        Returns:\n            CategorizedAnswerFrequencyLists. The domain object for categorized\n            answer frequency dict.\n        "
        return cls({category: AnswerFrequencyList.from_raw_type(answer_occurrence_list) for (category, answer_occurrence_list) in categorized_frequency_dict.items()})

class StateAnswersCalcOutput:
    """Domain object that represents output of calculations operating on
    state answers.
    """

    def __init__(self, exploration_id: str, exploration_version: int, state_name: str, interaction_id: str, calculation_id: str, calculation_output: Union[AnswerFrequencyList, CategorizedAnswerFrequencyLists]) -> None:
        if False:
            print('Hello World!')
        'Initialize domain object for state answers calculation output.\n\n        Args:\n            exploration_id: str. The ID of the exploration corresponding to the\n                answer calculation output.\n            exploration_version: int. The version of the exploration\n                corresponding to the answer calculation output.\n            state_name: str. The name of the exploration state to which the\n                aggregated answers were submitted.\n            interaction_id: str. The ID of the interaction.\n            calculation_id: str. Which calculation was performed on the given\n                answer data.\n            calculation_output: AnswerCalculationOutput. The output of an\n                answer aggregation operation.\n        '
        self.exploration_id = exploration_id
        self.exploration_version = exploration_version
        self.state_name = state_name
        self.calculation_id = calculation_id
        self.interaction_id = interaction_id
        self.calculation_output = calculation_output

    def validate(self) -> None:
        if False:
            while True:
                i = 10
        'Validates StateAnswersCalcOutputModel domain object entity before\n        it is commited to storage.\n        '
        max_bytes_per_calc_output_data = 999999
        if not isinstance(self.exploration_id, str):
            raise utils.ValidationError('Expected exploration_id to be a string, received %s' % str(self.exploration_id))
        if not isinstance(self.state_name, str):
            raise utils.ValidationError('Expected state_name to be a string, received %s' % str(self.state_name))
        if not isinstance(self.calculation_id, str):
            raise utils.ValidationError('Expected calculation_id to be a string, received %s' % str(self.calculation_id))
        if not isinstance(self.calculation_output, AnswerFrequencyList) and (not isinstance(self.calculation_output, CategorizedAnswerFrequencyLists)):
            raise utils.ValidationError('Expected calculation output to be one of AnswerFrequencyList or CategorizedAnswerFrequencyLists, encountered: %s' % self.calculation_output)
        output_data = self.calculation_output.to_raw_type()
        if sys.getsizeof(output_data) > max_bytes_per_calc_output_data:
            raise utils.ValidationError('calculation_output is too big to be stored (size: %d): %s' % (sys.getsizeof(output_data), str(output_data)))

class LearnerAnswerDetails:
    """Domain object that represents the answer details submitted by the
    learner.
    """

    def __init__(self, state_reference: str, entity_type: str, interaction_id: str, learner_answer_info_list: List[LearnerAnswerInfo], accumulated_answer_info_json_size_bytes: int, learner_answer_info_schema_version: int=feconf.CURRENT_LEARNER_ANSWER_INFO_SCHEMA_VERSION) -> None:
        if False:
            while True:
                i = 10
        "Constructs a LearnerAnswerDetail domain object.\n\n        Args:\n            state_reference: str. This field is used to refer to a state\n                in an exploration or question. For an exploration the value\n                will be equal to 'exp_id:state_name' & for question this will\n                be equal to 'question_id' only.\n            entity_type: str. The type of entity, for which the domain\n                object is being created. The value must be one of\n                ENTITY_TYPE_EXPLORATION or ENTITY_TYPE_QUESTION.\n            interaction_id: str. The ID of the interaction, but this value\n                should not be equal to EndExploration and\n                Continue as these interactions cannot solicit answer\n                details.\n            learner_answer_info_list: list(LearnerAnswerInfo). The list of\n                LearnerAnswerInfo objects.\n            accumulated_answer_info_json_size_bytes: int. The size of\n                learner_answer_info_list in bytes.\n            learner_answer_info_schema_version: int. The schema version of the\n                LearnerAnswerInfo dict.\n        "
        self.state_reference = state_reference
        self.entity_type = entity_type
        self.interaction_id = interaction_id
        self.learner_answer_info_list = learner_answer_info_list
        self.accumulated_answer_info_json_size_bytes = accumulated_answer_info_json_size_bytes
        self.learner_answer_info_schema_version = learner_answer_info_schema_version

    def to_dict(self) -> LearnerAnswerDetailsDict:
        if False:
            while True:
                i = 10
        'Returns a dict representing LearnerAnswerDetails domain object.\n\n        Returns:\n            dict. A dict, mapping all fields of LearnerAnswerDetails instance.\n        '
        return {'state_reference': self.state_reference, 'entity_type': self.entity_type, 'interaction_id': self.interaction_id, 'learner_answer_info_list': [learner_answer_info.to_dict() for learner_answer_info in self.learner_answer_info_list], 'accumulated_answer_info_json_size_bytes': self.accumulated_answer_info_json_size_bytes, 'learner_answer_info_schema_version': self.learner_answer_info_schema_version}

    @classmethod
    def from_dict(cls, learner_answer_details_dict: LearnerAnswerDetailsDict) -> LearnerAnswerDetails:
        if False:
            while True:
                i = 10
        'Return a LearnerAnswerDetails domain object from a dict.\n\n        Args:\n            learner_answer_details_dict: dict. The dict representation of\n                LearnerAnswerDetails object.\n\n        Returns:\n            LearnerAnswerDetails. The corresponding LearnerAnswerDetails\n            domain object.\n        '
        return cls(learner_answer_details_dict['state_reference'], learner_answer_details_dict['entity_type'], learner_answer_details_dict['interaction_id'], [LearnerAnswerInfo.from_dict(learner_answer_info_dict) for learner_answer_info_dict in learner_answer_details_dict['learner_answer_info_list']], learner_answer_details_dict['accumulated_answer_info_json_size_bytes'], learner_answer_details_dict['learner_answer_info_schema_version'])

    def validate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Validates LearnerAnswerDetails domain object.'
        if not isinstance(self.state_reference, str):
            raise utils.ValidationError('Expected state_reference to be a string, received %s' % str(self.state_reference))
        if not isinstance(self.entity_type, str):
            raise utils.ValidationError('Expected entity_type to be a string, received %s' % str(self.entity_type))
        split_state_reference = self.state_reference.split(':')
        if self.entity_type == feconf.ENTITY_TYPE_EXPLORATION:
            if len(split_state_reference) != 2:
                raise utils.ValidationError("For entity type exploration, the state reference should be of the form 'exp_id:state_name', but received %s" % self.state_reference)
        elif self.entity_type == feconf.ENTITY_TYPE_QUESTION:
            if len(split_state_reference) != 1:
                raise utils.ValidationError("For entity type question, the state reference should be of the form 'question_id', but received %s" % self.state_reference)
        else:
            raise utils.ValidationError('Invalid entity type received %s' % self.entity_type)
        if not isinstance(self.interaction_id, str):
            raise utils.ValidationError('Expected interaction_id to be a string, received %s' % str(self.interaction_id))
        if self.interaction_id not in interaction_registry.Registry.get_all_interaction_ids():
            raise utils.ValidationError('Unknown interaction_id: %s' % self.interaction_id)
        if self.interaction_id in constants.INTERACTION_IDS_WITHOUT_ANSWER_DETAILS:
            raise utils.ValidationError('The %s interaction does not support soliciting answer details from learners.' % self.interaction_id)
        if not isinstance(self.learner_answer_info_list, list):
            raise utils.ValidationError('Expected learner_answer_info_list to be a list, received %s' % str(self.learner_answer_info_list))
        for learner_answer_info in self.learner_answer_info_list:
            learner_answer_info.validate()
        if not isinstance(self.learner_answer_info_schema_version, int):
            raise utils.ValidationError('Expected learner_answer_info_schema_version to be an int, received %s' % self.learner_answer_info_schema_version)
        if not isinstance(self.accumulated_answer_info_json_size_bytes, int):
            raise utils.ValidationError('Expected accumulated_answer_info_json_size_bytes to be an int received %s' % self.accumulated_answer_info_json_size_bytes)

    def add_learner_answer_info(self, learner_answer_info: LearnerAnswerInfo) -> None:
        if False:
            return 10
        'Adds new learner answer info in the learner_answer_info_list.\n\n        Args:\n            learner_answer_info: LearnerAnswerInfo. The learner answer info\n                object, which is created after the learner has submitted the\n                details of the answer.\n        '
        learner_answer_info_dict_size = learner_answer_info.get_learner_answer_info_dict_size()
        if self.accumulated_answer_info_json_size_bytes + learner_answer_info_dict_size <= MAX_LEARNER_ANSWER_INFO_LIST_BYTE_SIZE:
            self.learner_answer_info_list.append(learner_answer_info)
            self.accumulated_answer_info_json_size_bytes += learner_answer_info_dict_size

    def delete_learner_answer_info(self, learner_answer_info_id: str) -> None:
        if False:
            return 10
        'Delete the learner answer info from the learner_answer_info_list.\n\n        Args:\n            learner_answer_info_id: str. The learner answer info\n                id, which needs to be deleted from\n                the learner_answer_info_list.\n\n        Raises:\n            Exception. If the learner answer info with the given id is not\n                found in the learner answer info list.\n        '
        new_learner_answer_info_list = []
        for learner_answer_info in self.learner_answer_info_list:
            if learner_answer_info.id != learner_answer_info_id:
                new_learner_answer_info_list.append(learner_answer_info)
            else:
                self.accumulated_answer_info_json_size_bytes -= learner_answer_info.get_learner_answer_info_dict_size()
        if self.learner_answer_info_list == new_learner_answer_info_list:
            raise Exception('Learner answer info with the given id not found.')
        self.learner_answer_info_list = new_learner_answer_info_list

    def update_state_reference(self, new_state_reference: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Updates the state_reference of the LearnerAnswerDetails object.\n\n        Args:\n            new_state_reference: str. The new state reference of the\n                LearnerAnswerDetails.\n        '
        self.state_reference = new_state_reference

class LearnerAnswerInfo:
    """Domain object containing the answer details submitted by the learner."""

    def __init__(self, learner_answer_info_id: str, answer: Optional[Union[str, int, Dict[str, str], List[str]]], answer_details: str, created_on: datetime.datetime) -> None:
        if False:
            return 10
        "Constructs a LearnerAnswerInfo domain object.\n\n        Args:\n            learner_answer_info_id: str. The id of the LearnerAnswerInfo object.\n            answer: dict or list or str or int or bool. The answer which is\n                submitted by the learner. Actually type of the answer is\n                interaction dependent, like TextInput interactions have\n                string type answer, NumericInput have int type answers etc.\n            answer_details: str. The details the learner will submit when the\n                learner will be asked questions like 'Hey how did you land on\n                this answer', 'Why did you pick that answer' etc.\n            created_on: datetime. The time at which the answer details were\n                received.\n        "
        self.id = learner_answer_info_id
        self.answer = answer
        self.answer_details = answer_details
        self.created_on = created_on

    def to_dict(self) -> LearnerAnswerInfoDict:
        if False:
            while True:
                i = 10
        'Returns the dict of learner answer info.\n\n        Returns:\n            dict. The learner_answer_info dict.\n        '
        learner_answer_info_dict: LearnerAnswerInfoDict = {'id': self.id, 'answer': self.answer, 'answer_details': self.answer_details, 'created_on': self.created_on.strftime('%Y-%m-%d %H:%M:%S.%f')}
        return learner_answer_info_dict

    @classmethod
    def from_dict(cls, learner_answer_info_dict: LearnerAnswerInfoDict) -> LearnerAnswerInfo:
        if False:
            print('Hello World!')
        'Returns a dict representing LearnerAnswerInfo domain object.\n\n        Returns:\n            dict. A dict, mapping all fields of LearnerAnswerInfo instance.\n        '
        return cls(learner_answer_info_dict['id'], learner_answer_info_dict['answer'], learner_answer_info_dict['answer_details'], datetime.datetime.strptime(learner_answer_info_dict['created_on'], '%Y-%m-%d %H:%M:%S.%f'))

    @classmethod
    def get_new_learner_answer_info_id(cls) -> str:
        if False:
            print('Hello World!')
        'Generates the learner answer info domain object id.\n\n        Returns:\n            learner_answer_info_id: str. The id generated by the function.\n        '
        learner_answer_info_id = utils.base64_from_int(int(utils.get_current_time_in_millisecs())) + utils.base64_from_int(utils.get_random_int(127 * 127))
        return learner_answer_info_id

    def validate(self) -> None:
        if False:
            return 10
        'Validates the LearnerAnswerInfo domain object.'
        if not isinstance(self.id, str):
            raise utils.ValidationError('Expected id to be a string, received %s' % self.id)
        if self.answer is None:
            raise utils.ValidationError('The answer submitted by the learner cannot be empty')
        if isinstance(self.answer, dict):
            if self.answer == {}:
                raise utils.ValidationError('The answer submitted cannot be an empty dict.')
        if isinstance(self.answer, str):
            if self.answer == '':
                raise utils.ValidationError('The answer submitted cannot be an empty string')
        if not isinstance(self.answer_details, str):
            raise utils.ValidationError('Expected answer_details to be a string, received %s' % type(self.answer_details))
        if self.answer_details == '':
            raise utils.ValidationError('The answer details submitted cannot be an empty string.')
        if sys.getsizeof(self.answer_details) > MAX_ANSWER_DETAILS_BYTE_SIZE:
            raise utils.ValidationError('The answer details size is to large to be stored')
        if not isinstance(self.created_on, datetime.datetime):
            raise utils.ValidationError('Expected created_on to be a datetime, received %s' % str(self.created_on))

    def get_learner_answer_info_dict_size(self) -> int:
        if False:
            i = 10
            return i + 15
        'Returns a size overestimate (in bytes) of the given learner answer\n        info dict.\n\n        Returns:\n            int. Size of the learner_answer_info_dict in bytes.\n        '
        learner_answer_info_dict = self.to_dict()
        return sys.getsizeof(json.dumps(learner_answer_info_dict, default=str))