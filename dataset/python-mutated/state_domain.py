"""Domain object for states and their constituents."""
from __future__ import annotations
import copy
import itertools
import logging
import math
import re
from core import android_validation_constants
from core import feconf
from core import schema_utils
from core import utils
from core.constants import constants
from core.domain import customization_args_util
from core.domain import param_domain
from core.domain import translation_domain
from extensions import domain
from extensions.objects.models import objects
from typing import Any, Callable, Dict, Iterator, List, Literal, Mapping, Optional, Tuple, Type, TypedDict, TypeVar, Union, cast, overload
from core.domain import html_cleaner
from core.domain import interaction_registry
from core.domain import rules_registry
MYPY = False
if MYPY:
    from extensions.interactions import base
_GenericCustomizationArgType = TypeVar('_GenericCustomizationArgType')
AllowedRuleSpecInputTypes = Union[str, int, float, List[str], List[List[str]], Mapping[str, Union[str, List[str], int, bool, float, Dict[str, int], List[Any]]]]

class TrainingDataDict(TypedDict):
    """Type for the training data dictionary."""
    answer_group_index: int
    answers: List[str]

class AnswerGroupDict(TypedDict):
    """Dictionary representing the AnswerGroup object."""
    outcome: OutcomeDict
    rule_specs: List[RuleSpecDict]
    training_data: List[str]
    tagged_skill_misconception_id: Optional[str]

class StateVersionHistoryDict(TypedDict):
    """Dictionary representing the StateVersionHistory object."""
    previously_edited_in_version: Optional[int]
    state_name_in_previous_version: Optional[str]
    committer_id: str
AcceptableCorrectAnswerTypes = Union[List[List[str]], List[str], str, Dict[str, str], int, None]

class AnswerGroup(translation_domain.BaseTranslatableObject):
    """Value object for an answer group. Answer groups represent a set of rules
    dictating whether a shared feedback should be shared with the user. These
    rules are ORed together. Answer groups may also support a classifier
    that involve soft matching of answers to a set of training data and/or
    example answers dictated by the creator.
    """

    def __init__(self, outcome: Outcome, rule_specs: List[RuleSpec], training_data: List[str], tagged_skill_misconception_id: Optional[str]) -> None:
        if False:
            i = 10
            return i + 15
        "Initializes a AnswerGroup domain object.\n\n        Args:\n            outcome: Outcome. The outcome corresponding to the answer group.\n            rule_specs: list(RuleSpec). List of rule specifications.\n            training_data: list(*). List of answers belonging to training\n                data of this answer group.\n            tagged_skill_misconception_id: str or None. The format is\n                '<skill_id>-<misconception_id>', where skill_id is the skill ID\n                of the tagged misconception and misconception_id is the id of\n                the tagged misconception for the answer group. It is not None\n                only when a state is part of a Question object that\n                tests a particular skill.\n        "
        self.rule_specs = [RuleSpec(rule_spec.rule_type, rule_spec.inputs) for rule_spec in rule_specs]
        self.outcome = outcome
        self.training_data = training_data
        self.tagged_skill_misconception_id = tagged_skill_misconception_id

    def get_translatable_contents_collection(self, **kwargs: Optional[str]) -> translation_domain.TranslatableContentsCollection:
        if False:
            i = 10
            return i + 15
        'Get all translatable fields in the answer group.\n\n        Returns:\n            translatable_contents_collection: TranslatableContentsCollection.\n            An instance of TranslatableContentsCollection class.\n        '
        translatable_contents_collection = translation_domain.TranslatableContentsCollection()
        if self.outcome is not None:
            translatable_contents_collection.add_fields_from_translatable_object(self.outcome)
        for rule_spec in self.rule_specs:
            if kwargs['interaction_id'] not in ['TextInput', 'SetInput']:
                break
            translatable_contents_collection.add_fields_from_translatable_object(rule_spec, interaction_id=kwargs['interaction_id'])
        return translatable_contents_collection

    def to_dict(self) -> AnswerGroupDict:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict representing this AnswerGroup domain object.\n\n        Returns:\n            dict. A dict, mapping all fields of AnswerGroup instance.\n        '
        return {'rule_specs': [rule_spec.to_dict() for rule_spec in self.rule_specs], 'outcome': self.outcome.to_dict(), 'training_data': self.training_data, 'tagged_skill_misconception_id': self.tagged_skill_misconception_id}

    @classmethod
    def from_dict(cls, answer_group_dict: AnswerGroupDict, validate: bool=True) -> AnswerGroup:
        if False:
            print('Hello World!')
        'Return a AnswerGroup domain object from a dict.\n\n        Args:\n            answer_group_dict: dict. The dict representation of AnswerGroup\n                object.\n            validate: bool. False, when the validations should not be called.\n\n        Returns:\n            AnswerGroup. The corresponding AnswerGroup domain object.\n        '
        return cls(Outcome.from_dict(answer_group_dict['outcome'], validate=validate), [RuleSpec.from_dict(rs) for rs in answer_group_dict['rule_specs']], answer_group_dict['training_data'], answer_group_dict['tagged_skill_misconception_id'])

    def validate(self, interaction: base.BaseInteraction, exp_param_specs_dict: Dict[str, param_domain.ParamSpec], *, tagged_skill_misconception_id_required: bool=False) -> None:
        if False:
            return 10
        "Verifies that all rule classes are valid, and that the AnswerGroup\n        only has one classifier rule.\n\n        Args:\n            interaction: BaseInteraction. The interaction object.\n            exp_param_specs_dict: dict. A dict of all parameters used in the\n                exploration. Keys are parameter names and values are ParamSpec\n                value objects with an object type property (obj_type).\n            tagged_skill_misconception_id_required: bool. The 'tagged_skill_\n                misconception_id' is required or not.\n\n        Raises:\n            ValidationError. One or more attributes of the AnswerGroup are\n                invalid.\n            ValidationError. The AnswerGroup contains more than one classifier\n                rule.\n            ValidationError. The tagged_skill_misconception_id is not valid.\n        "
        if not isinstance(self.rule_specs, list):
            raise utils.ValidationError('Expected answer group rules to be a list, received %s' % self.rule_specs)
        if self.tagged_skill_misconception_id is not None and (not tagged_skill_misconception_id_required):
            raise utils.ValidationError('Expected tagged skill misconception id to be None, received %s' % self.tagged_skill_misconception_id)
        if self.tagged_skill_misconception_id is not None and tagged_skill_misconception_id_required:
            if not isinstance(self.tagged_skill_misconception_id, str):
                raise utils.ValidationError('Expected tagged skill misconception id to be a str, received %s' % self.tagged_skill_misconception_id)
            if not re.match(constants.VALID_SKILL_MISCONCEPTION_ID_REGEX, self.tagged_skill_misconception_id):
                raise utils.ValidationError('Expected the format of tagged skill misconception id to be <skill_id>-<misconception_id>, received %s' % self.tagged_skill_misconception_id)
        if len(self.rule_specs) == 0:
            raise utils.ValidationError('There must be at least one rule for each answer group.')
        for rule_spec in self.rule_specs:
            if rule_spec.rule_type not in interaction.rules_dict:
                raise utils.ValidationError('Unrecognized rule type: %s' % rule_spec.rule_type)
            rule_spec.validate(interaction.get_rule_param_list(rule_spec.rule_type), exp_param_specs_dict)
        self.outcome.validate()

    @staticmethod
    def convert_html_in_answer_group(answer_group_dict: AnswerGroupDict, conversion_fn: Callable[[str], str], html_field_types_to_rule_specs: Dict[str, rules_registry.RuleSpecsExtensionDict]) -> AnswerGroupDict:
        if False:
            for i in range(10):
                print('nop')
        'Checks for HTML fields in an answer group dict and converts it\n        according to the conversion function.\n\n        Args:\n            answer_group_dict: dict. The answer group dict.\n            conversion_fn: function. The function to be used for converting the\n                HTML.\n            html_field_types_to_rule_specs: dict. A dictionary that specifies\n                the locations of html fields in rule specs. It is defined as a\n                mapping of rule input types to a dictionary containing\n                interaction id, format, and rule types. See\n                html_field_types_to_rule_specs_state_v41.json for an example.\n\n        Returns:\n            dict. The converted answer group dict.\n        '
        answer_group_dict['outcome']['feedback']['html'] = conversion_fn(answer_group_dict['outcome']['feedback']['html'])
        for (rule_spec_index, rule_spec) in enumerate(answer_group_dict['rule_specs']):
            answer_group_dict['rule_specs'][rule_spec_index] = RuleSpec.convert_html_in_rule_spec(rule_spec, conversion_fn, html_field_types_to_rule_specs)
        return answer_group_dict

class HintDict(TypedDict):
    """Dictionary representing the Hint object."""
    hint_content: SubtitledHtmlDict

class Hint(translation_domain.BaseTranslatableObject):
    """Value object representing a hint."""

    def __init__(self, hint_content: SubtitledHtml) -> None:
        if False:
            return 10
        'Constructs a Hint domain object.\n\n        Args:\n            hint_content: SubtitledHtml. The hint text and ID referring to the\n                other assets for this content.\n        '
        self.hint_content = hint_content

    def get_translatable_contents_collection(self, **kwargs: Optional[str]) -> translation_domain.TranslatableContentsCollection:
        if False:
            print('Hello World!')
        'Get all translatable fields in the hint.\n\n        Returns:\n            translatable_contents_collection: TranslatableContentsCollection.\n            An instance of TranslatableContentsCollection class.\n        '
        translatable_contents_collection = translation_domain.TranslatableContentsCollection()
        translatable_contents_collection.add_translatable_field(self.hint_content.content_id, translation_domain.ContentType.HINT, translation_domain.TranslatableContentFormat.HTML, self.hint_content.html)
        return translatable_contents_collection

    def to_dict(self) -> HintDict:
        if False:
            while True:
                i = 10
        'Returns a dict representing this Hint domain object.\n\n        Returns:\n            dict. A dict mapping the field of Hint instance.\n        '
        return {'hint_content': self.hint_content.to_dict()}

    @classmethod
    def from_dict(cls, hint_dict: HintDict, validate: bool=True) -> Hint:
        if False:
            return 10
        'Return a Hint domain object from a dict.\n\n        Args:\n            hint_dict: dict. The dict representation of Hint object.\n            validate: bool. False, when the validations should not be called.\n\n        Returns:\n            Hint. The corresponding Hint domain object.\n        '
        hint_content = SubtitledHtml.from_dict(hint_dict['hint_content'])
        if validate:
            hint_content.validate()
        return cls(hint_content)

    def validate(self) -> None:
        if False:
            print('Hello World!')
        'Validates all properties of Hint.'
        self.hint_content.validate()

    @staticmethod
    def convert_html_in_hint(hint_dict: HintDict, conversion_fn: Callable[[str], str]) -> HintDict:
        if False:
            while True:
                i = 10
        'Checks for HTML fields in the hints and converts it\n        according to the conversion function.\n\n        Args:\n            hint_dict: dict. The hints dict.\n            conversion_fn: function. The function to be used for converting the\n                HTML.\n\n        Returns:\n            dict. The converted hints dict.\n        '
        hint_dict['hint_content']['html'] = conversion_fn(hint_dict['hint_content']['html'])
        return hint_dict

class SolutionDict(TypedDict):
    """Dictionary representing the Solution object."""
    answer_is_exclusive: bool
    correct_answer: AcceptableCorrectAnswerTypes
    explanation: SubtitledHtmlDict

class Solution(translation_domain.BaseTranslatableObject):
    """Value object representing a solution.

    A solution consists of answer_is_exclusive, correct_answer and an
    explanation.When answer_is_exclusive is True, this indicates that it is
    the only correct answer; when it is False, this indicates that it is one
    possible answer. correct_answer records an answer that enables the learner
    to progress to the next card and explanation is an HTML string containing
    an explanation for the solution.
    """

    def __init__(self, interaction_id: str, answer_is_exclusive: bool, correct_answer: AcceptableCorrectAnswerTypes, explanation: SubtitledHtml) -> None:
        if False:
            i = 10
            return i + 15
        "Constructs a Solution domain object.\n\n        Args:\n            interaction_id: str. The interaction id.\n            answer_is_exclusive: bool. True if is the only correct answer;\n                False if is one of possible answer.\n            correct_answer: *. The correct answer; this answer\n                enables the learner to progress to the next card. The type of\n                correct_answer is determined by the value of\n                BaseInteraction.answer_type. Some examples for the types are\n                list(set(str)), list(str), str, dict(str, str), etc.\n            explanation: SubtitledHtml. Contains text and text id to link audio\n                translations for the solution's explanation.\n        "
        self.answer_is_exclusive = answer_is_exclusive
        self.correct_answer = interaction_registry.Registry.get_interaction_by_id(interaction_id).normalize_answer(correct_answer)
        self.explanation = explanation

    def get_translatable_contents_collection(self, **kwargs: Optional[str]) -> translation_domain.TranslatableContentsCollection:
        if False:
            print('Hello World!')
        'Get all translatable fields in the solution.\n\n        Returns:\n            translatable_contents_collection: TranslatableContentsCollection.\n            An instance of TranslatableContentsCollection class.\n        '
        translatable_contents_collection = translation_domain.TranslatableContentsCollection()
        translatable_contents_collection.add_translatable_field(self.explanation.content_id, translation_domain.ContentType.SOLUTION, translation_domain.TranslatableContentFormat.HTML, self.explanation.html)
        return translatable_contents_collection

    def to_dict(self) -> SolutionDict:
        if False:
            i = 10
            return i + 15
        'Returns a dict representing this Solution domain object.\n\n        Returns:\n            dict. A dict mapping all fields of Solution instance.\n        '
        return {'answer_is_exclusive': self.answer_is_exclusive, 'correct_answer': self.correct_answer, 'explanation': self.explanation.to_dict()}

    @classmethod
    def from_dict(cls, interaction_id: str, solution_dict: SolutionDict, validate: bool=True) -> Solution:
        if False:
            while True:
                i = 10
        'Return a Solution domain object from a dict.\n\n        Args:\n            interaction_id: str. The interaction id.\n            solution_dict: dict. The dict representation of Solution object.\n            validate: bool. False, when the validations should not be called.\n\n        Returns:\n            Solution. The corresponding Solution domain object.\n        '
        explanation = SubtitledHtml.from_dict(solution_dict['explanation'])
        if validate:
            explanation.validate()
        return cls(interaction_id, solution_dict['answer_is_exclusive'], interaction_registry.Registry.get_interaction_by_id(interaction_id).normalize_answer(solution_dict['correct_answer']), explanation)

    def validate(self, interaction_id: str) -> None:
        if False:
            i = 10
            return i + 15
        'Validates all properties of Solution.\n\n        Args:\n            interaction_id: str. The interaction id.\n\n        Raises:\n            ValidationError. One or more attributes of the Solution are not\n                valid.\n        '
        if not isinstance(self.answer_is_exclusive, bool):
            raise utils.ValidationError('Expected answer_is_exclusive to be bool, received %s' % self.answer_is_exclusive)
        interaction_registry.Registry.get_interaction_by_id(interaction_id).normalize_answer(self.correct_answer)
        self.explanation.validate()

    @staticmethod
    def convert_html_in_solution(interaction_id: Optional[str], solution_dict: SolutionDict, conversion_fn: Callable[[str], str], html_field_types_to_rule_specs: Dict[str, rules_registry.RuleSpecsExtensionDict], interaction_spec: base.BaseInteractionDict) -> SolutionDict:
        if False:
            i = 10
            return i + 15
        'Checks for HTML fields in a solution and convert it according\n        to the conversion function.\n\n        Args:\n            interaction_id: Optional[str]. The interaction id.\n            solution_dict: dict. The Solution dict.\n            conversion_fn: function. The function to be used for converting the\n                HTML.\n            html_field_types_to_rule_specs: dict. A dictionary that specifies\n                the locations of html fields in rule specs. It is defined as a\n                mapping of rule input types to a dictionary containing\n                interaction id, format, and rule types. See\n                html_field_types_to_rule_specs_state_v41.json for an example.\n            interaction_spec: dict. The specification for the interaction.\n\n        Returns:\n            dict. The converted Solution dict.\n\n        Raises:\n            Exception. The Solution dict has an invalid answer type.\n        '
        if interaction_id is None:
            return solution_dict
        solution_dict['explanation']['html'] = conversion_fn(solution_dict['explanation']['html'])
        if interaction_spec['can_have_solution']:
            if solution_dict['correct_answer']:
                for html_type in html_field_types_to_rule_specs.keys():
                    if html_type == interaction_spec['answer_type']:
                        if html_type == feconf.ANSWER_TYPE_LIST_OF_SETS_OF_HTML:
                            assert isinstance(solution_dict['correct_answer'], list)
                            for (list_index, html_list) in enumerate(solution_dict['correct_answer']):
                                assert isinstance(html_list, list)
                                for (answer_html_index, answer_html) in enumerate(html_list):
                                    correct_answer = cast(List[List[str]], solution_dict['correct_answer'])
                                    correct_answer[list_index][answer_html_index] = conversion_fn(answer_html)
                        elif html_type == feconf.ANSWER_TYPE_SET_OF_HTML:
                            assert isinstance(solution_dict['correct_answer'], list)
                            for (answer_html_index, answer_html) in enumerate(solution_dict['correct_answer']):
                                assert isinstance(answer_html, str)
                                set_of_html_correct_answer = cast(List[str], solution_dict['correct_answer'])
                                set_of_html_correct_answer[answer_html_index] = conversion_fn(answer_html)
                        else:
                            raise Exception('The solution does not have a valid correct_answer type.')
        return solution_dict

class InteractionInstanceDict(TypedDict):
    """Dictionary representing the InteractionInstance object."""
    id: Optional[str]
    customization_args: CustomizationArgsDictType
    answer_groups: List[AnswerGroupDict]
    default_outcome: Optional[OutcomeDict]
    confirmed_unclassified_answers: List[AnswerGroup]
    hints: List[HintDict]
    solution: Optional[SolutionDict]

class InteractionInstance(translation_domain.BaseTranslatableObject):
    """Value object for an instance of an interaction."""

    class RangeVariableDict(TypedDict):
        """Dictionary representing the range variable for the NumericInput
        interaction.
        """
        ans_group_index: int
        rule_spec_index: int
        lower_bound: Optional[float]
        upper_bound: Optional[float]
        lb_inclusive: bool
        ub_inclusive: bool

    class MatchedDenominatorDict(TypedDict):
        """Dictionary representing the matched denominator variable for the
        FractionInput interaction.
        """
        ans_group_index: int
        rule_spec_index: int
        denominator: int
    _DEFAULT_INTERACTION_ID = None

    def __init__(self, interaction_id: Optional[str], customization_args: Dict[str, InteractionCustomizationArg], answer_groups: List[AnswerGroup], default_outcome: Optional[Outcome], confirmed_unclassified_answers: List[AnswerGroup], hints: List[Hint], solution: Optional[Solution]) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Initializes a InteractionInstance domain object.\n\n        Args:\n            interaction_id: Optional[str]. The interaction id.\n            customization_args: dict. The customization dict. The keys are\n                names of customization_args and the values are dicts with a\n                single key, 'value', whose corresponding value is the value of\n                the customization arg.\n            answer_groups: list(AnswerGroup). List of answer groups of the\n                interaction instance.\n            default_outcome: Optional[Outcome]. The default outcome of the\n                interaction instance, or None if no default outcome exists\n                for the interaction.\n            confirmed_unclassified_answers: list(*). List of answers which have\n                been confirmed to be associated with the default outcome.\n            hints: list(Hint). List of hints for this interaction.\n            solution: Solution|None. A possible solution for the question asked\n                in this interaction, or None if no solution exists for the\n                interaction.\n        "
        self.id = interaction_id
        self.customization_args = customization_args
        self.answer_groups = answer_groups
        self.default_outcome = default_outcome
        self.confirmed_unclassified_answers = confirmed_unclassified_answers
        self.hints = hints
        self.solution = solution

    def get_translatable_contents_collection(self, **kwargs: Optional[str]) -> translation_domain.TranslatableContentsCollection:
        if False:
            for i in range(10):
                print('nop')
        'Get all translatable fields in the interaction instance.\n\n        Returns:\n            translatable_contents_collection: TranslatableContentsCollection.\n            An instance of TranslatableContentsCollection class.\n        '
        translatable_contents_collection = translation_domain.TranslatableContentsCollection()
        if self.default_outcome is not None:
            translatable_contents_collection.add_fields_from_translatable_object(self.default_outcome)
        for answer_group in self.answer_groups:
            translatable_contents_collection.add_fields_from_translatable_object(answer_group, interaction_id=self.id)
        for customization_arg in self.customization_args.values():
            translatable_contents_collection.add_fields_from_translatable_object(customization_arg, interaction_id=self.id)
        for hint in self.hints:
            translatable_contents_collection.add_fields_from_translatable_object(hint)
        if self.solution is not None:
            translatable_contents_collection.add_fields_from_translatable_object(self.solution)
        return translatable_contents_collection

    def to_dict(self) -> InteractionInstanceDict:
        if False:
            print('Hello World!')
        'Returns a dict representing this InteractionInstance domain object.\n\n        Returns:\n            dict. A dict mapping all fields of InteractionInstance instance.\n        '
        customization_args_dict = {}
        if self.id:
            for ca_name in self.customization_args:
                customization_args_dict[ca_name] = self.customization_args[ca_name].to_customization_arg_dict()
        return {'id': self.id, 'customization_args': customization_args_dict, 'answer_groups': [group.to_dict() for group in self.answer_groups], 'default_outcome': self.default_outcome.to_dict() if self.default_outcome is not None else None, 'confirmed_unclassified_answers': self.confirmed_unclassified_answers, 'hints': [hint.to_dict() for hint in self.hints], 'solution': self.solution.to_dict() if self.solution else None}

    @classmethod
    def from_dict(cls, interaction_dict: InteractionInstanceDict, validate: bool=True) -> InteractionInstance:
        if False:
            i = 10
            return i + 15
        'Return a InteractionInstance domain object from a dict.\n\n        Args:\n            interaction_dict: dict. The dict representation of\n                InteractionInstance object.\n            validate: bool. False, when the validations should not be called.\n\n        Returns:\n            InteractionInstance. The corresponding InteractionInstance domain\n            object.\n        '
        default_outcome_dict = Outcome.from_dict(interaction_dict['default_outcome'], validate=validate) if interaction_dict['default_outcome'] is not None else None
        solution_dict = Solution.from_dict(interaction_dict['id'], interaction_dict['solution'], validate=validate) if interaction_dict['solution'] is not None and interaction_dict['id'] is not None else None
        customization_args = InteractionInstance.convert_customization_args_dict_to_customization_args(interaction_dict['id'], interaction_dict['customization_args'])
        return cls(interaction_dict['id'], customization_args, [AnswerGroup.from_dict(h, validate=validate) for h in interaction_dict['answer_groups']], default_outcome_dict, interaction_dict['confirmed_unclassified_answers'], [Hint.from_dict(h, validate=validate) for h in interaction_dict['hints']], solution_dict)

    @property
    def is_terminal(self) -> bool:
        if False:
            return 10
        'Determines if this interaction type is terminal. If no ID is set for\n        this interaction, it is assumed to not be terminal.\n\n        Returns:\n            bool. Whether the interaction is terminal.\n        '
        return bool(self.id and interaction_registry.Registry.get_interaction_by_id(self.id).is_terminal)

    @property
    def is_linear(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Determines if this interaction type is linear.\n\n        Returns:\n            bool. Whether the interaction is linear.\n        '
        return interaction_registry.Registry.get_interaction_by_id(self.id).is_linear

    def is_supported_on_android_app(self) -> bool:
        if False:
            return 10
        'Determines whether the interaction is a valid interaction that is\n        supported by the Android app.\n\n        Returns:\n            bool. Whether the interaction is supported by the Android app.\n        '
        return self.id is None or self.id in android_validation_constants.VALID_INTERACTION_IDS

    def is_rte_content_supported_on_android(self, require_valid_component_names: Callable[[str], bool]) -> bool:
        if False:
            print('Hello World!')
        'Determines whether the RTE content in interaction answer groups,\n        hints and solution is supported by Android app.\n\n        Args:\n            require_valid_component_names: function. Function to check\n                whether the RTE tags in the html string are allowed.\n\n        Returns:\n            bool. Whether the RTE content is valid.\n        '
        for answer_group in self.answer_groups:
            if require_valid_component_names(answer_group.outcome.feedback.html):
                return False
        if self.default_outcome and self.default_outcome.feedback and require_valid_component_names(self.default_outcome.feedback.html):
            return False
        for hint in self.hints:
            if require_valid_component_names(hint.hint_content.html):
                return False
        if self.solution and self.solution.explanation and require_valid_component_names(self.solution.explanation.html):
            return False
        return True

    def get_all_outcomes(self) -> List[Outcome]:
        if False:
            return 10
        'Returns a list of all outcomes of this interaction, taking into\n        consideration every answer group and the default outcome.\n\n        Returns:\n            list(Outcome). List of all outcomes of this interaction.\n        '
        outcomes = []
        for answer_group in self.answer_groups:
            outcomes.append(answer_group.outcome)
        if self.default_outcome is not None:
            outcomes.append(self.default_outcome)
        return outcomes

    def _validate_continue_interaction(self) -> None:
        if False:
            print('Hello World!')
        'Validates Continue interaction.'
        button_text_subtitled_unicode = cast(SubtitledUnicode, self.customization_args['buttonText'].value)
        text_value = button_text_subtitled_unicode.unicode_str
        if len(text_value) > 20:
            raise utils.ValidationError('The `continue` interaction text length should be atmost 20 characters.')

    def _validate_end_interaction(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Validates End interaction.'
        recc_exp_ids = cast(List[str], self.customization_args['recommendedExplorationIds'].value)
        if len(recc_exp_ids) > 3:
            raise utils.ValidationError('The total number of recommended explorations inside End interaction should be atmost 3.')

    def _validates_choices_should_be_unique_and_nonempty(self, choices: List[SubtitledHtml]) -> None:
        if False:
            return 10
        'Validates that the choices should be unique and non empty.\n\n        Args:\n            choices: List[state_domain.SubtitledHtml]. Choices that needs to\n                be validated.\n\n        Raises:\n            utils.ValidationError. Choice is empty.\n            utils.ValidationError. Choice is duplicate.\n        '
        seen_choices = []
        for choice in choices:
            if html_cleaner.is_html_empty(choice.html):
                raise utils.ValidationError('Choices should be non empty.')
            if choice.html not in seen_choices:
                seen_choices.append(choice.html)
            else:
                raise utils.ValidationError('Choices should be unique.')

    def _set_lower_and_upper_bounds(self, range_var: RangeVariableDict, lower_bound: float, upper_bound: float, *, lb_inclusive: bool, ub_inclusive: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Sets the lower and upper bounds for the range_var.\n\n        Args:\n            range_var: RangeVariableDict. Variable used to keep track of each\n                range.\n            lower_bound: float. The lower bound.\n            upper_bound: float. The upper bound.\n            lb_inclusive: bool. If lower bound is inclusive.\n            ub_inclusive: bool. If upper bound is inclusive.\n        '
        range_var['lower_bound'] = lower_bound
        range_var['upper_bound'] = upper_bound
        range_var['lb_inclusive'] = lb_inclusive
        range_var['ub_inclusive'] = ub_inclusive

    def _is_enclosed_by(self, test_range: RangeVariableDict, base_range: RangeVariableDict) -> bool:
        if False:
            while True:
                i = 10
        'Returns `True` when `test_range` variable lies within\n        `base_range` variable.\n\n        Args:\n            test_range: RangeVariableDictDict. It represents the variable for\n                which we have to check the range.\n            base_range: RangeVariableDictDict. It is the variable to which\n                the range is compared.\n\n        Returns:\n            bool. Returns True if test_range lies\n            within base_range.\n        '
        if base_range['lower_bound'] is None or test_range['lower_bound'] is None or base_range['upper_bound'] is None or (test_range['upper_bound'] is None):
            return False
        lb_satisfied = base_range['lower_bound'] < test_range['lower_bound'] or (base_range['lower_bound'] == test_range['lower_bound'] and (not test_range['lb_inclusive'] or base_range['lb_inclusive']))
        ub_satisfied = base_range['upper_bound'] > test_range['upper_bound'] or (base_range['upper_bound'] == test_range['upper_bound'] and (not test_range['ub_inclusive'] or base_range['ub_inclusive']))
        return lb_satisfied and ub_satisfied

    def _should_check_range_criteria(self, earlier_rule: RuleSpec, later_rule: RuleSpec) -> bool:
        if False:
            while True:
                i = 10
        'Compares the rule types of two rule specs to determine whether\n        to check for range enclosure.\n\n        Args:\n            earlier_rule: RuleSpec. Previous rule.\n            later_rule: RuleSpec. Current rule.\n\n        Returns:\n            bool. Returns True if the rules passes the range criteria check.\n        '
        if earlier_rule.rule_type in ('HasDenominatorEqualTo', 'IsEquivalentTo', 'IsLessThan', 'IsEquivalentToAndInSimplestForm', 'IsGreaterThan'):
            return True
        return later_rule.rule_type in ('HasDenominatorEqualTo', 'IsLessThan', 'IsGreaterThan')

    def _get_rule_value_of_fraction_interaction(self, rule_spec: RuleSpec) -> float:
        if False:
            while True:
                i = 10
        "Returns rule value of the rule_spec of FractionInput interaction so\n        that we can keep track of rule's range.\n\n        Args:\n            rule_spec: RuleSpec. Rule spec of an answer group.\n\n        Returns:\n            rule_value_f: float. The value of the rule spec.\n        "
        rule_value_f = rule_spec.inputs['f']
        value: float = rule_value_f['wholeNumber'] + float(rule_value_f['numerator']) / rule_value_f['denominator']
        return value

    def _validate_numeric_input(self, strict: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Validates the NumericInput interaction.\n\n        Args:\n            strict: bool. If True, the exploration is assumed to be published.\n\n        Raises:\n            ValidationError. Duplicate rules are present.\n            ValidationError. Rule having a solution that is subset of previous\n                rules' solution.\n            ValidationError. The 'tol' value in 'IsWithinTolerance' is negetive.\n            ValidationError. The 'a' is greater than or equal to 'b' in\n                'IsInclusivelyBetween' rule.\n        "
        lower_infinity = float('-inf')
        upper_infinity = float('inf')
        ranges: List[InteractionInstance.RangeVariableDict] = []
        rule_spec_till_now: List[RuleSpecDict] = []
        for (ans_group_index, answer_group) in enumerate(self.answer_groups):
            for (rule_spec_index, rule_spec) in enumerate(answer_group.rule_specs):
                if rule_spec.to_dict() in rule_spec_till_now and strict:
                    raise utils.ValidationError(f"The rule '{rule_spec_index}' of answer group '{ans_group_index}' of NumericInput interaction is already present.")
                rule_spec_till_now.append(rule_spec.to_dict())
                range_var: InteractionInstance.RangeVariableDict = {'ans_group_index': int(ans_group_index), 'rule_spec_index': int(rule_spec_index), 'lower_bound': None, 'upper_bound': None, 'lb_inclusive': False, 'ub_inclusive': False}
                if rule_spec.rule_type == 'IsLessThanOrEqualTo':
                    rule_value = float(rule_spec.inputs['x'])
                    self._set_lower_and_upper_bounds(range_var, lower_infinity, rule_value, lb_inclusive=False, ub_inclusive=True)
                elif rule_spec.rule_type == 'IsGreaterThanOrEqualTo':
                    rule_value = float(rule_spec.inputs['x'])
                    self._set_lower_and_upper_bounds(range_var, rule_value, upper_infinity, lb_inclusive=True, ub_inclusive=False)
                elif rule_spec.rule_type == 'Equals':
                    rule_value = float(rule_spec.inputs['x'])
                    self._set_lower_and_upper_bounds(range_var, rule_value, rule_value, lb_inclusive=True, ub_inclusive=True)
                elif rule_spec.rule_type == 'IsLessThan':
                    rule_value = float(rule_spec.inputs['x'])
                    self._set_lower_and_upper_bounds(range_var, lower_infinity, rule_value, lb_inclusive=False, ub_inclusive=False)
                elif rule_spec.rule_type == 'IsGreaterThan':
                    rule_value = float(rule_spec.inputs['x'])
                    self._set_lower_and_upper_bounds(range_var, rule_value, upper_infinity, lb_inclusive=False, ub_inclusive=False)
                elif rule_spec.rule_type == 'IsWithinTolerance':
                    rule_value_x = float(rule_spec.inputs['x'])
                    rule_value_tol = float(rule_spec.inputs['tol'])
                    if rule_value_tol <= 0.0:
                        raise utils.ValidationError(f"The rule '{rule_spec_index}' of answer group '{ans_group_index}' having rule type 'IsWithinTolerance' have 'tol' value less than or equal to zero in NumericInput interaction.")
                    self._set_lower_and_upper_bounds(range_var, rule_value_x - rule_value_tol, rule_value_x + rule_value_tol, lb_inclusive=True, ub_inclusive=True)
                elif rule_spec.rule_type == 'IsInclusivelyBetween':
                    rule_value_a = float(rule_spec.inputs['a'])
                    rule_value_b = float(rule_spec.inputs['b'])
                    if rule_value_a >= rule_value_b and strict:
                        raise utils.ValidationError(f"The rule '{rule_spec_index}' of answer group '{ans_group_index}' having rule type 'IsInclusivelyBetween' have `a` value greater than `b` value in NumericInput interaction.")
                    self._set_lower_and_upper_bounds(range_var, rule_value_a, rule_value_b, lb_inclusive=True, ub_inclusive=True)
                for range_ele in ranges:
                    if self._is_enclosed_by(range_var, range_ele) and strict:
                        raise utils.ValidationError(f"Rule '{rule_spec_index}' from answer group '{ans_group_index}' will never be matched because it is made redundant by the above rules")
                ranges.append(range_var)

    def _validate_fraction_input(self, strict: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        "Validates the FractionInput interaction.\n\n        Args:\n            strict: bool. If True, the exploration is assumed to be published.\n\n        Raises:\n            ValidationError. Duplicate rules are present.\n            ValidationError. Solution is not in simplest form when the\n                'simplest form' setting is turned on.\n            ValidationError. Solution is not in proper form, having values\n                like 1 2/3 when the 'proper form' setting is turned on.\n            ValidationError. Solution is not in proper form, when the 'proper\n                form' setting is turned on.\n            ValidationError. The 'IsExactlyEqualTo' rule have integral value\n                when 'allow non zero integers' setting is off.\n            ValidationError. Rule have solution that is subset of previous\n                rules' solutions.\n            ValidationError. The 'HasFractionalPartExactlyEqualTo' rule comes\n                after 'HasDenominatorEqualTo' rule where the fractional\n                denominator is equal to 'HasDenominatorEqualTo' rule value.\n        "
        ranges: List[InteractionInstance.RangeVariableDict] = []
        matched_denominator_list: List[InteractionInstance.MatchedDenominatorDict] = []
        rule_spec_till_now: List[RuleSpecDict] = []
        inputs_without_fractions = ['HasDenominatorEqualTo', 'HasNumeratorEqualTo', 'HasIntegerPartEqualTo', 'HasNoFractionalPart']
        rules_that_can_have_improper_fractions = ['IsExactlyEqualTo', 'HasFractionalPartExactlyEqualTo']
        lower_infinity = float('-inf')
        upper_infinity = float('inf')
        allow_non_zero_integ_part = self.customization_args['allowNonzeroIntegerPart'].value
        allow_imp_frac = self.customization_args['allowImproperFraction'].value
        require_simple_form = self.customization_args['requireSimplestForm'].value
        for (ans_group_index, answer_group) in enumerate(self.answer_groups):
            for (rule_spec_index, rule_spec) in enumerate(answer_group.rule_specs):
                if rule_spec.to_dict() in rule_spec_till_now and strict:
                    raise utils.ValidationError(f"The rule '{rule_spec_index}' of answer group '{ans_group_index}' of FractionInput interaction is already present.")
                rule_spec_till_now.append(rule_spec.to_dict())
                if rule_spec.rule_type not in inputs_without_fractions:
                    num = rule_spec.inputs['f']['numerator']
                    den = rule_spec.inputs['f']['denominator']
                    whole = rule_spec.inputs['f']['wholeNumber']
                    if require_simple_form and strict:
                        d = math.gcd(num, den)
                        val_num = num // d
                        val_den = den // d
                        if val_num != num and val_den != den:
                            raise utils.ValidationError(f"The rule '{rule_spec_index}' of answer group '{ans_group_index}' do not have value in simple form in FractionInput interaction.")
                    if strict and (not allow_imp_frac) and (den <= num) and (rule_spec.rule_type in rules_that_can_have_improper_fractions):
                        raise utils.ValidationError(f"The rule '{rule_spec_index}' of answer group '{ans_group_index}' do not have value in proper fraction in FractionInput interaction.")
                range_var: InteractionInstance.RangeVariableDict = {'ans_group_index': int(ans_group_index), 'rule_spec_index': int(rule_spec_index), 'lower_bound': None, 'upper_bound': None, 'lb_inclusive': False, 'ub_inclusive': False}
                matched_denominator: InteractionInstance.MatchedDenominatorDict = {'ans_group_index': int(ans_group_index), 'rule_spec_index': int(rule_spec_index), 'denominator': 0}
                if rule_spec.rule_type in ('IsEquivalentTo', 'IsExactlyEqualTo', 'IsEquivalentToAndInSimplestForm'):
                    if rule_spec.rule_type == 'IsExactlyEqualTo' and (not allow_non_zero_integ_part) and (whole != 0) and strict:
                        raise utils.ValidationError(f"The rule '{rule_spec_index}' of answer group '{ans_group_index}' has non zero integer part in FractionInput interaction.")
                    rule_value_f = self._get_rule_value_of_fraction_interaction(rule_spec)
                    self._set_lower_and_upper_bounds(range_var, rule_value_f, rule_value_f, lb_inclusive=True, ub_inclusive=True)
                if rule_spec.rule_type == 'IsGreaterThan':
                    rule_value_f = self._get_rule_value_of_fraction_interaction(rule_spec)
                    self._set_lower_and_upper_bounds(range_var, rule_value_f, upper_infinity, lb_inclusive=False, ub_inclusive=False)
                if rule_spec.rule_type == 'IsLessThan':
                    rule_value_f = self._get_rule_value_of_fraction_interaction(rule_spec)
                    self._set_lower_and_upper_bounds(range_var, lower_infinity, rule_value_f, lb_inclusive=False, ub_inclusive=False)
                if rule_spec.rule_type == 'HasDenominatorEqualTo':
                    rule_value_x = int(rule_spec.inputs['x'])
                    matched_denominator['denominator'] = rule_value_x
                for range_ele in ranges:
                    earlier_rule = self.answer_groups[range_ele['ans_group_index']].rule_specs[range_ele['rule_spec_index']]
                    if self._should_check_range_criteria(earlier_rule, rule_spec) and self._is_enclosed_by(range_var, range_ele) and strict:
                        raise utils.ValidationError(f"Rule '{rule_spec_index}' from answer group '{ans_group_index}' of FractionInput interaction will never be matched because it is made redundant by the above rules")
                for den in matched_denominator_list:
                    if den is not None and rule_spec.rule_type == 'HasFractionalPartExactlyEqualTo' and (den['denominator'] == rule_spec.inputs['f']['denominator']):
                        raise utils.ValidationError(f"Rule '{rule_spec_index}' from answer group '{ans_group_index}' of FractionInput interaction having rule type HasFractionalPartExactlyEqualTo will never be matched because it is made redundant by the above rules")
                ranges.append(range_var)
                matched_denominator_list.append(matched_denominator)

    def _validate_number_with_units_input(self, strict: bool=False) -> None:
        if False:
            print('Hello World!')
        "Validates the NumberWithUnitsInput interaction.\n\n        Args:\n            strict: bool. If True, the exploration is assumed to be published.\n\n        Raises:\n            ValidationError. Duplicate rules are present.\n            ValidationError. The 'IsEqualTo' rule comes after 'IsEquivalentTo'\n                rule having same values.\n        "
        number_with_units_rules = []
        rule_spec_till_now: List[RuleSpecDict] = []
        for (ans_group_index, answer_group) in enumerate(self.answer_groups):
            for (rule_spec_index, rule_spec) in enumerate(answer_group.rule_specs):
                if rule_spec.to_dict() in rule_spec_till_now and strict:
                    raise utils.ValidationError(f"The rule '{rule_spec_index}' of answer group '{ans_group_index}' of NumberWithUnitsInput interaction is already present.")
                rule_spec_till_now.append(rule_spec.to_dict())
                if rule_spec.rule_type == 'IsEquivalentTo':
                    number_with_units_rules.append(rule_spec.inputs['f'])
                if rule_spec.rule_type == 'IsEqualTo' and rule_spec.inputs['f'] in number_with_units_rules and strict:
                    raise utils.ValidationError(f"The rule '{rule_spec_index}' of answer group '{ans_group_index}' has rule type equal is coming after rule type equivalent having same value in FractionInput interaction.")

    def _validate_multi_choice_input(self, strict: bool=False) -> None:
        if False:
            return 10
        'Validates the MultipleChoiceInput interaction.\n\n        Args:\n            strict: bool. If True, the exploration is assumed to be published.\n\n        Raises:\n            ValidationError. Duplicate rules are present.\n            ValidationError. Answer choices are empty or duplicate.\n        '
        rule_spec_till_now: List[RuleSpecDict] = []
        choices = cast(List[SubtitledHtml], self.customization_args['choices'].value)
        self._validates_choices_should_be_unique_and_nonempty(choices)
        for (ans_group_index, answer_group) in enumerate(self.answer_groups):
            for (rule_spec_index, rule_spec) in enumerate(answer_group.rule_specs):
                if rule_spec.to_dict() in rule_spec_till_now and strict:
                    raise utils.ValidationError(f"The rule '{rule_spec_index}' of answer group '{ans_group_index}' of MultipleChoiceInput interaction is already present.")
                rule_spec_till_now.append(rule_spec.to_dict())

    def _validate_item_selec_input(self, strict: bool=False) -> None:
        if False:
            while True:
                i = 10
        "Validates the ItemSelectionInput interaction.\n\n        Args:\n            strict: bool. If True, the exploration is assumed to be published.\n\n        Raises:\n            ValidationError. Duplicate rules are present.\n            ValidationError. The 'Equals' rule does not have value between min\n                and max number of selections.\n            ValidationError. Minimum number of selections value is greater\n                than maximum number of selections value.\n            ValidationError. Not enough choices to have minimum number of\n                selections.\n            ValidationError. Answer choices are empty or duplicate.\n        "
        min_value = cast(int, self.customization_args['minAllowableSelectionCount'].value)
        max_value = cast(int, self.customization_args['maxAllowableSelectionCount'].value)
        rule_spec_till_now: List[RuleSpecDict] = []
        choices = cast(List[SubtitledHtml], self.customization_args['choices'].value)
        self._validates_choices_should_be_unique_and_nonempty(choices)
        if min_value > max_value:
            raise utils.ValidationError(f'Min value which is {str(min_value)} is greater than max value which is {str(max_value)} in ItemSelectionInput interaction.')
        if len(choices) < min_value:
            raise utils.ValidationError(f'Number of choices which is {str(len(choices))} is lesser than the min value selection which is {str(min_value)} in ItemSelectionInput interaction.')
        for (ans_group_index, answer_group) in enumerate(self.answer_groups):
            for (rule_spec_index, rule_spec) in enumerate(answer_group.rule_specs):
                if rule_spec.to_dict() in rule_spec_till_now and strict:
                    raise utils.ValidationError(f'The rule {rule_spec_index} of answer group {ans_group_index} of ItemSelectionInput interaction is already present.')
                rule_spec_till_now.append(rule_spec.to_dict())
                if rule_spec.rule_type == 'Equals':
                    if strict and (len(rule_spec.inputs['x']) < min_value or len(rule_spec.inputs['x']) > max_value):
                        raise utils.ValidationError(f"Selected choices of rule '{rule_spec_index}' of answer group '{ans_group_index}' either less than min_selection_value or greater than max_selection_value in ItemSelectionInput interaction.")

    def _validate_drag_and_drop_input(self, strict: bool=False) -> None:
        if False:
            print('Hello World!')
        "Validates the DragAndDropInput interaction.\n\n        Args:\n            strict: bool. If True, the exploration is assumed to be published.\n\n        Raises:\n            ValidationError. Duplicate rules are present.\n            ValidationError. Multiple items at the same place when the setting\n                is turned off.\n            ValidationError. The 'IsEqualToOrderingWithOneItemAtIncorrect\n                Position' rule present when 'multiple items at same place'\n                setting turned off.\n            ValidationError. In 'HasElementXBeforeElementY' rule, 'X' value\n                is equal to 'Y' value.\n            ValidationError. The 'IsEqualToOrdering' rule have empty values.\n            ValidationError. The 'IsEqualToOrdering' rule comes after\n                'HasElementXAtPositionY' where element 'X' is present at\n                position 'Y' in 'IsEqualToOrdering' rule.\n            ValidationError. Less than 2 items are present.\n            ValidationError. Answer choices are empty or duplicate.\n        "
        multi_item_value = self.customization_args['allowMultipleItemsInSamePosition'].value
        ele_x_at_y_rules = []
        rule_spec_till_now: List[RuleSpecDict] = []
        equal_ordering_one_at_incorec_posn = []
        choices = cast(List[SubtitledHtml], self.customization_args['choices'].value)
        if len(choices) < 2:
            raise utils.ValidationError('There should be atleast 2 values inside DragAndDrop interaction.')
        self._validates_choices_should_be_unique_and_nonempty(choices)
        for (ans_group_index, answer_group) in enumerate(self.answer_groups):
            for (rule_spec_index, rule_spec) in enumerate(answer_group.rule_specs):
                if rule_spec.to_dict() in rule_spec_till_now and strict:
                    raise utils.ValidationError(f"The rule '{rule_spec_index}' of answer group '{ans_group_index}' of DragAndDropInput interaction is already present.")
                rule_spec_till_now.append(rule_spec.to_dict())
                if strict and (not multi_item_value) and (rule_spec.rule_type == 'IsEqualToOrderingWithOneItemAtIncorrectPosition'):
                    raise utils.ValidationError(f"The rule '{rule_spec_index}' of answer group '{ans_group_index}' having rule type - IsEqualToOrderingWithOneItemAtIncorrectPosition should not be there when the multiple items in same position setting is turned off in DragAndDropSortInput interaction.")
                if not multi_item_value and strict:
                    for ele in rule_spec.inputs['x']:
                        if len(ele) > 1:
                            raise utils.ValidationError(f"The rule '{rule_spec_index}' of answer group '{ans_group_index}' have multiple items at same place when multiple items in same position settings is turned off in DragAndDropSortInput interaction.")
                if rule_spec.rule_type == 'HasElementXBeforeElementY' and rule_spec.inputs['x'] == rule_spec.inputs['y'] and strict:
                    raise utils.ValidationError(f"The rule '{rule_spec_index}' of answer group '{ans_group_index}', the value 1 and value 2 cannot be same when rule type is HasElementXBeforeElementY of DragAndDropSortInput interaction.")
                if rule_spec.rule_type == 'HasElementXAtPositionY':
                    element = rule_spec.inputs['x']
                    position = rule_spec.inputs['y']
                    ele_x_at_y_rules.append({'element': element, 'position': position})
                if rule_spec.rule_type == 'IsEqualToOrderingWithOneItemAtIncorrectPosition':
                    equal_ordering_one_at_incorec_posn.append(rule_spec.inputs['x'])
                if rule_spec.rule_type == 'IsEqualToOrdering':
                    if len(rule_spec.inputs['x']) <= 0:
                        raise utils.ValidationError(f"The rule '{rule_spec_index}'of answer group '{ans_group_index}', having rule type IsEqualToOrdering should not have empty values.")
                    if strict:
                        for ele in ele_x_at_y_rules:
                            ele_position = ele['position']
                            ele_element = ele['element']
                            if ele_position > len(rule_spec.inputs['x']):
                                continue
                            rule_choice = rule_spec.inputs['x'][ele_position - 1]
                            for choice in rule_choice:
                                if choice == ele_element:
                                    raise utils.ValidationError(f'Rule - {rule_spec_index} of answer group {ans_group_index} will never be match because it is made redundant by the HasElementXAtPositionY rule above.')
                        item_to_layer_idx = {}
                        for (layer_idx, layer) in enumerate(rule_spec.inputs['x']):
                            for item in layer:
                                item_to_layer_idx[item] = layer_idx
                        for ele in equal_ordering_one_at_incorec_posn:
                            wrong_positions = 0
                            for (layer_idx, layer) in enumerate(ele):
                                for item in layer:
                                    if layer_idx != item_to_layer_idx[item]:
                                        wrong_positions += 1
                            if wrong_positions <= 1:
                                raise utils.ValidationError(f'Rule - {rule_spec_index} of answer group {ans_group_index} will never be match because it is made redundant by the IsEqualToOrderingWithOneItemAtIncorrectPosition rule above.')

    def _validate_text_input(self, strict: bool=False) -> None:
        if False:
            print('Hello World!')
        "Validates the TextInput interaction.\n\n        Args:\n            strict: bool. If True, the exploration is assumed to be published.\n\n        Raises:\n            ValidationError. Text input height is not >= 1 and <= 10.\n            ValidationError. Duplicate rules are present.\n            ValidationError. The 'Contains' rule comes before another 'Contains'\n                rule, where 'Contains' rule string is a substring of other\n                rules string.\n            ValidationError. The 'Contains' rule comes before 'StartsWith'\n                rule, where 'Contains' rule string is a substring of other\n                rules string.\n            ValidationError. The 'Contains' rule comes before 'Equals'\n                rule, where 'Contains' rule string is a substring of other\n                rules string.\n            ValidationError. The 'StartsWith' rule comes before the 'Equals'\n                rule where the 'StartsWith' rule string is a prefix of other\n                rules string.\n            ValidationError. The 'StartsWith' rule comes before the another\n                'StartsWith' rule where the 'StartsWith' rule string is\n                a prefix of other rules string.\n        "
        rule_spec_till_now: List[RuleSpecDict] = []
        seen_strings_contains: List[List[str]] = []
        seen_strings_startswith: List[List[str]] = []
        rows_value = cast(int, self.customization_args['rows'].value)
        if rows_value < 1 or rows_value > 10:
            raise utils.ValidationError('Rows value in Text interaction should be between 1 and 10.')
        for (ans_group_idx, answer_group) in enumerate(self.answer_groups):
            for (rule_spec_idx, rule_spec) in enumerate(answer_group.rule_specs):
                if rule_spec.to_dict() in rule_spec_till_now and strict:
                    raise utils.ValidationError(f"The rule '{rule_spec_idx}' of answer group '{ans_group_idx}' of TextInput interaction is already present.")
                rule_spec_till_now.append(rule_spec.to_dict())
                if rule_spec.rule_type == 'Contains':
                    if not strict:
                        continue
                    rule_values = rule_spec.inputs['x']['normalizedStrSet']
                    for contain_rule_ele in seen_strings_contains:
                        for contain_rule_string in contain_rule_ele:
                            for rule_value in rule_values:
                                if contain_rule_string in rule_value:
                                    raise utils.ValidationError(f"Rule - '{rule_spec_idx}' of answer group - '{ans_group_idx}' having rule type '{rule_spec.rule_type}' will never be matched because it is made redundant by the above 'contains' rule.")
                    seen_strings_contains.append(rule_spec.inputs['x']['normalizedStrSet'])
                if rule_spec.rule_type == 'StartsWith':
                    if not strict:
                        continue
                    rule_values = rule_spec.inputs['x']['normalizedStrSet']
                    for start_with_rule_ele in seen_strings_startswith:
                        for start_with_rule_string in start_with_rule_ele:
                            for rule_value in rule_values:
                                if rule_value.startswith(start_with_rule_string):
                                    raise utils.ValidationError(f"Rule - '{rule_spec_idx}' of answer group - '{ans_group_idx}' having rule type '{rule_spec.rule_type}' will never be matched because it is made redundant by the above 'StartsWith' rule.")
                    for contain_rule_ele in seen_strings_contains:
                        for contain_rule_string in contain_rule_ele:
                            for rule_value in rule_values:
                                if contain_rule_string in rule_value:
                                    raise utils.ValidationError(f"Rule - '{rule_spec_idx}' of answer group - '{ans_group_idx}' having rule type '{rule_spec.rule_type}' will never be matched because it is made redundant by the above 'contains' rule.")
                    seen_strings_startswith.append(rule_values)
                if rule_spec.rule_type == 'Equals':
                    if not strict:
                        continue
                    rule_values = rule_spec.inputs['x']['normalizedStrSet']
                    for contain_rule_ele in seen_strings_contains:
                        for contain_rule_string in contain_rule_ele:
                            for rule_value in rule_values:
                                if contain_rule_string in rule_value:
                                    raise utils.ValidationError(f"Rule - '{rule_spec_idx}' of answer group - '{ans_group_idx}' having rule type '{rule_spec.rule_type}' will never be matched because it is made redundant by the above 'contains' rule.")
                    for start_with_rule_ele in seen_strings_startswith:
                        for start_with_rule_string in start_with_rule_ele:
                            for rule_value in rule_values:
                                if rule_value.startswith(start_with_rule_string):
                                    raise utils.ValidationError(f"Rule - '{rule_spec_idx}' of answer group - '{ans_group_idx}' having rule type '{rule_spec.rule_type}' will never be matched because it is made redundant by the above 'StartsWith' rule.")

    def validate(self, exp_param_specs_dict: Dict[str, param_domain.ParamSpec], *, tagged_skill_misconception_id_required: bool=False, strict: bool=False) -> None:
        if False:
            print('Hello World!')
        "Validates various properties of the InteractionInstance.\n\n        Args:\n            exp_param_specs_dict: dict. A dict of specified parameters used in\n                the exploration. Keys are parameter names and values are\n                ParamSpec value objects with an object type property(obj_type).\n                Is used to validate AnswerGroup objects.\n            tagged_skill_misconception_id_required: bool. The 'tagged_skill_\n                misconception_id' is required or not.\n            strict: bool. Tells if the validation is strict or not.\n\n        Raises:\n            ValidationError. One or more attributes of the InteractionInstance\n                are invalid.\n        "
        if not isinstance(self.id, str):
            raise utils.ValidationError('Expected interaction id to be a string, received %s' % self.id)
        try:
            interaction = interaction_registry.Registry.get_interaction_by_id(self.id)
        except KeyError as e:
            raise utils.ValidationError('Invalid interaction id: %s' % self.id) from e
        self._validate_customization_args()
        if not isinstance(self.answer_groups, list):
            raise utils.ValidationError('Expected answer groups to be a list, received %s.' % self.answer_groups)
        if not self.is_terminal and self.default_outcome is None:
            raise utils.ValidationError('Non-terminal interactions must have a default outcome.')
        if self.is_terminal and self.default_outcome is not None:
            raise utils.ValidationError('Terminal interactions must not have a default outcome.')
        if self.is_terminal and self.answer_groups:
            raise utils.ValidationError('Terminal interactions must not have any answer groups.')
        if self.is_linear and self.answer_groups:
            raise utils.ValidationError('Linear interactions must not have any answer groups.')
        for answer_group in self.answer_groups:
            answer_group.validate(interaction, exp_param_specs_dict, tagged_skill_misconception_id_required=tagged_skill_misconception_id_required)
        if self.default_outcome is not None:
            self.default_outcome.validate()
        if not isinstance(self.hints, list):
            raise utils.ValidationError('Expected hints to be a list, received %s' % self.hints)
        for hint in self.hints:
            hint.validate()
        if self.solution:
            self.solution.validate(self.id)
        interaction_id_to_strict_validation_func = {'NumericInput': self._validate_numeric_input, 'FractionInput': self._validate_fraction_input, 'NumberWithUnits': self._validate_number_with_units_input, 'MultipleChoiceInput': self._validate_multi_choice_input, 'ItemSelectionInput': self._validate_item_selec_input, 'DragAndDropSortInput': self._validate_drag_and_drop_input, 'TextInput': self._validate_text_input}
        interaction_id_to_non_strict_validation_func = {'Continue': self._validate_continue_interaction, 'EndExploration': self._validate_end_interaction}
        if self.id in interaction_id_to_strict_validation_func:
            interaction_id_to_strict_validation_func[self.id](strict)
        elif self.id in interaction_id_to_non_strict_validation_func:
            interaction_id_to_non_strict_validation_func[self.id]()

    def _validate_customization_args(self) -> None:
        if False:
            while True:
                i = 10
        'Validates the customization arguments keys and values using\n        customization_args_util.validate_customization_args_and_values().\n        '
        if not isinstance(self.customization_args, dict):
            raise utils.ValidationError('Expected customization args to be a dict, received %s' % self.customization_args)
        customization_args_dict = {}
        if self.id:
            for ca_name in self.customization_args:
                try:
                    customization_args_dict[ca_name] = self.customization_args[ca_name].to_customization_arg_dict()
                except AttributeError as e:
                    raise utils.ValidationError('Expected customization arg value to be a InteractionCustomizationArg domain object, received %s' % self.customization_args[ca_name]) from e
        assert self.id is not None
        interaction = interaction_registry.Registry.get_interaction_by_id(self.id)
        customization_args_util.validate_customization_args_and_values('interaction', self.id, customization_args_dict, interaction.customization_arg_specs)
        self.customization_args = InteractionInstance.convert_customization_args_dict_to_customization_args(self.id, customization_args_dict)

    @classmethod
    def create_default_interaction(cls, default_dest_state_name: Optional[str], content_id_for_default_outcome: str) -> InteractionInstance:
        if False:
            for i in range(10):
                print('nop')
        "Create a default InteractionInstance domain object:\n            - customization_args: empty dictionary;\n            - answer_groups: empty list;\n            - default_outcome: dest is set to 'default_dest_state_name' and\n                feedback and param_changes are initialized as empty lists;\n            - confirmed_unclassified_answers: empty list;\n\n        Args:\n            default_dest_state_name: str|None. The default destination state, or\n                None if no default destination is provided.\n            content_id_for_default_outcome: str. The content id for the default\n                outcome.\n\n        Returns:\n            InteractionInstance. The corresponding InteractionInstance domain\n            object with default values.\n        "
        default_outcome = Outcome(default_dest_state_name, None, SubtitledHtml.create_default_subtitled_html(content_id_for_default_outcome), False, [], None, None)
        return cls(cls._DEFAULT_INTERACTION_ID, {}, [], default_outcome, [], [], None)

    @staticmethod
    def convert_html_in_interaction(interaction_dict: InteractionInstanceDict, ca_specs_dict: List[domain.CustomizationArgSpecsDict], conversion_fn: Callable[[str], str]) -> InteractionInstanceDict:
        if False:
            for i in range(10):
                print('nop')
        'Checks for HTML fields in the interaction and converts it\n        according to the conversion function.\n\n        Args:\n            interaction_dict: dict. The interaction dict.\n            ca_specs_dict: dict. The customization args dict.\n            conversion_fn: function. The function to be used for converting the\n                HTML.\n\n        Returns:\n            dict. The converted interaction dict.\n        '

        def wrapped_conversion_fn(value: SubtitledHtml, schema_obj_type: str) -> SubtitledHtml:
            if False:
                for i in range(10):
                    print('nop')
            "Applies the conversion function to the SubtitledHtml values.\n\n            Args:\n                value: SubtitledHtml|SubtitledUnicode. The value in the\n                    customization argument value to be converted.\n                schema_obj_type: str. The schema obj_type for the customization\n                    argument value, which is one of 'SubtitledUnicode' or\n                    'SubtitledHtml'.\n\n            Returns:\n                SubtitledHtml|SubtitledUnicode. The converted SubtitledHtml\n                object, if schema_type is 'SubititledHtml', otherwise the\n                unmodified SubtitledUnicode object.\n            "
            if schema_obj_type == schema_utils.SCHEMA_OBJ_TYPE_SUBTITLED_HTML:
                value.html = conversion_fn(value.html)
            return value
        customization_args = InteractionCustomizationArg.convert_cust_args_dict_to_cust_args_based_on_specs(interaction_dict['customization_args'], ca_specs_dict)
        for ca_spec in ca_specs_dict:
            ca_spec_name = ca_spec['name']
            customization_args[ca_spec_name].value = InteractionCustomizationArg.traverse_by_schema_and_convert(ca_spec['schema'], customization_args[ca_spec_name].value, wrapped_conversion_fn)
        customization_args_dict = {}
        for ca_name in customization_args:
            customization_args_dict[ca_name] = customization_args[ca_name].to_customization_arg_dict()
        interaction_dict['customization_args'] = customization_args_dict
        return interaction_dict

    @staticmethod
    def convert_customization_args_dict_to_customization_args(interaction_id: Optional[str], customization_args_dict: CustomizationArgsDictType, state_schema_version: int=feconf.CURRENT_STATE_SCHEMA_VERSION) -> Dict[str, InteractionCustomizationArg]:
        if False:
            for i in range(10):
                print('nop')
        "Converts customization arguments dictionary to customization\n        arguments. This is done by converting each customization argument to a\n        InteractionCustomizationArg domain object.\n\n        Args:\n            interaction_id: str. The interaction id.\n            customization_args_dict: dict. A dictionary of customization\n                argument name to a customization argument dict, which is a dict\n                of the single key 'value' to the value of the customization\n                argument.\n            state_schema_version: int. The state schema version.\n\n        Returns:\n            dict. A dictionary of customization argument names to the\n            InteractionCustomizationArg domain object's.\n        "
        all_interaction_ids = interaction_registry.Registry.get_all_interaction_ids()
        interaction_id_is_valid = interaction_id not in all_interaction_ids
        if interaction_id_is_valid or interaction_id is None:
            return {}
        ca_specs_dict = interaction_registry.Registry.get_all_specs_for_state_schema_version(state_schema_version, can_fetch_latest_specs=True)[interaction_id]['customization_arg_specs']
        return InteractionCustomizationArg.convert_cust_args_dict_to_cust_args_based_on_specs(customization_args_dict, ca_specs_dict)

class InteractionCustomizationArg(translation_domain.BaseTranslatableObject):
    """Object representing an interaction's customization argument.
    Any SubtitledHtml or SubtitledUnicode values in the customization argument
    value are represented as their respective domain objects here, rather than a
    SubtitledHtml dict or SubtitledUnicode dict.
    """

    def __init__(self, value: UnionOfCustomizationArgsDictValues, schema: Dict[str, Union[SubtitledHtmlDict, SubtitledUnicodeDict, str]]) -> None:
        if False:
            while True:
                i = 10
        'Initializes a InteractionCustomizationArg domain object.\n\n        Args:\n            value: *. The value of the interaction customization argument.\n            schema: dict. The schema defining the specification of the value.\n        '
        self.value = value
        self.schema = schema

    def get_translatable_contents_collection(self, **kwargs: Optional[str]) -> translation_domain.TranslatableContentsCollection:
        if False:
            i = 10
            return i + 15
        'Get all translatable fields in the interaction customization args.\n\n        Returns:\n            translatable_contents_collection: TranslatableContentsCollection.\n            An instance of TranslatableContentsCollection class.\n        '
        translatable_contents_collection = translation_domain.TranslatableContentsCollection()
        subtitled_htmls = self.get_subtitled_html()
        for subtitled_html in subtitled_htmls:
            translatable_contents_collection.add_translatable_field(subtitled_html.content_id, translation_domain.ContentType.CUSTOMIZATION_ARG, translation_domain.TranslatableContentFormat.HTML, subtitled_html.html, kwargs['interaction_id'])
        subtitled_unicodes = self.get_subtitled_unicode()
        for subtitled_unicode in subtitled_unicodes:
            translatable_contents_collection.add_translatable_field(subtitled_unicode.content_id, translation_domain.ContentType.CUSTOMIZATION_ARG, translation_domain.TranslatableContentFormat.UNICODE_STRING, subtitled_unicode.unicode_str, kwargs['interaction_id'])
        return translatable_contents_collection

    def to_customization_arg_dict(self) -> Dict[str, UnionOfCustomizationArgsDictValues]:
        if False:
            i = 10
            return i + 15
        'Converts a InteractionCustomizationArgument domain object to a\n        customization argument dictionary. This is done by\n        traversing the customization argument schema, and converting\n        SubtitledUnicode to unicode and SubtitledHtml to html where appropriate.\n        '

        @overload
        def convert_content_to_dict(ca_value: SubtitledHtml, unused_schema_obj_type: str) -> SubtitledHtmlDict:
            if False:
                for i in range(10):
                    print('nop')
            ...

        @overload
        def convert_content_to_dict(ca_value: SubtitledUnicode, unused_schema_obj_type: str) -> SubtitledUnicodeDict:
            if False:
                return 10
            ...

        def convert_content_to_dict(ca_value: Union[SubtitledHtml, SubtitledUnicode], unused_schema_obj_type: str) -> Union[SubtitledHtmlDict, SubtitledUnicodeDict]:
            if False:
                return 10
            "Conversion function used to convert SubtitledHtml to\n            SubtitledHtml dicts and SubtitledUnicode to SubtitledUnicode dicts.\n\n            Args:\n                ca_value: SubtitledHtml|SubtitledUnicode. A SubtitledUnicode or\n                    SubtitledHtml value found inside the customization\n                    argument value.\n                unused_schema_obj_type: str. The schema obj_type for the\n                    customization argument value, which is one\n                    of 'SubtitledUnicode' or 'SubtitledHtml'.\n\n            Returns:\n                dict. The customization argument value converted to a dict.\n            "
            return ca_value.to_dict()
        return {'value': InteractionCustomizationArg.traverse_by_schema_and_convert(self.schema, copy.deepcopy(self.value), convert_content_to_dict)}

    @classmethod
    def from_customization_arg_dict(cls, ca_dict: Dict[str, UnionOfCustomizationArgsDictValues], ca_schema: Dict[str, Any]) -> InteractionCustomizationArg:
        if False:
            while True:
                i = 10
        "Converts a customization argument dictionary to an\n        InteractionCustomizationArgument domain object. This is done by\n        traversing the customization argument schema, and converting\n        unicode to SubtitledUnicode and html to SubtitledHtml where appropriate.\n\n        Args:\n            ca_dict: dict. The customization argument dictionary. A dict of the\n                single key 'value' to the value of the customization argument.\n            ca_schema: dict. The schema that defines the customization argument\n                value.\n\n        Returns:\n            InteractionCustomizationArg. The customization argument domain\n            object.\n        "

        @overload
        def convert_content_to_domain_obj(ca_value: Dict[str, str], schema_obj_type: Literal['SubtitledUnicode']) -> SubtitledUnicode:
            if False:
                i = 10
                return i + 15
            ...

        @overload
        def convert_content_to_domain_obj(ca_value: Dict[str, str], schema_obj_type: Literal['SubtitledHtml']) -> SubtitledHtml:
            if False:
                i = 10
                return i + 15
            ...

        def convert_content_to_domain_obj(ca_value: Dict[str, str], schema_obj_type: str) -> Union[SubtitledHtml, SubtitledUnicode]:
            if False:
                while True:
                    i = 10
            "Conversion function used to convert SubtitledHtml dicts to\n            SubtitledHtml and SubtitledUnicode dicts to SubtitledUnicode.\n\n            Args:\n                ca_value: dict. Value of customization argument.\n                schema_obj_type: str. The schema obj_type for the customization\n                    argument value, which is one of 'SubtitledUnicode' or\n                    'SubtitledHtml'.\n\n            Returns:\n                dict. The unmodified customization argument value.\n            "
            if schema_obj_type == schema_utils.SCHEMA_OBJ_TYPE_SUBTITLED_UNICODE:
                class_obj: Union[SubtitledUnicode, SubtitledHtml] = SubtitledUnicode(ca_value['content_id'], ca_value['unicode_str'])
            if schema_obj_type == schema_utils.SCHEMA_OBJ_TYPE_SUBTITLED_HTML:
                class_obj = SubtitledHtml(ca_value['content_id'], ca_value['html'])
            return class_obj
        ca_value = InteractionCustomizationArg.traverse_by_schema_and_convert(ca_schema, copy.deepcopy(ca_dict['value']), convert_content_to_domain_obj)
        return cls(ca_value, ca_schema)

    def get_subtitled_unicode(self) -> List[SubtitledUnicode]:
        if False:
            for i in range(10):
                print('nop')
        'Get all SubtitledUnicode(s) in the customization argument.\n\n        Returns:\n            list(SubtitledUnicode). A list of SubtitledUnicode.\n        '
        return InteractionCustomizationArg.traverse_by_schema_and_get(self.schema, self.value, [schema_utils.SCHEMA_OBJ_TYPE_SUBTITLED_UNICODE], lambda x: x)

    def get_subtitled_html(self) -> List[SubtitledHtml]:
        if False:
            print('Hello World!')
        'Get all SubtitledHtml(s) in the customization argument.\n\n        Returns:\n            list(SubtitledHtml). A list of SubtitledHtml.\n        '
        return InteractionCustomizationArg.traverse_by_schema_and_get(self.schema, self.value, [schema_utils.SCHEMA_OBJ_TYPE_SUBTITLED_HTML], lambda x: x)

    def get_content_ids(self) -> List[str]:
        if False:
            while True:
                i = 10
        'Get all content_ids from SubtitledHtml and SubtitledUnicode in the\n        customization argument.\n\n        Returns:\n            list(str). A list of content_ids.\n        '
        return InteractionCustomizationArg.traverse_by_schema_and_get(self.schema, self.value, [schema_utils.SCHEMA_OBJ_TYPE_SUBTITLED_UNICODE, schema_utils.SCHEMA_OBJ_TYPE_SUBTITLED_HTML], lambda x: x.content_id)

    def validate_subtitled_html(self) -> None:
        if False:
            while True:
                i = 10
        'Calls the validate method on all SubtitledHtml domain objects in\n        the customization arguments.\n        '

        def validate_html(subtitled_html: SubtitledHtml) -> None:
            if False:
                for i in range(10):
                    print('nop')
            'A dummy value extractor that calls the validate method on\n            the passed SubtitledHtml domain object.\n            '
            subtitled_html.validate()
        InteractionCustomizationArg.traverse_by_schema_and_get(self.schema, self.value, [schema_utils.SCHEMA_OBJ_TYPE_SUBTITLED_HTML], validate_html)

    @staticmethod
    def traverse_by_schema_and_convert(schema: Dict[str, Any], value: _GenericCustomizationArgType, conversion_fn: AcceptableConversionFnType) -> _GenericCustomizationArgType:
        if False:
            while True:
                i = 10
        "Helper function that recursively traverses an interaction\n        customization argument spec to locate any SubtitledHtml or\n        SubtitledUnicode objects, and applies a conversion function to the\n        customization argument value.\n\n        Args:\n            schema: dict. The customization dict to be modified: dict\n                with a single key, 'value', whose corresponding value is the\n                value of the customization arg.\n            value: dict. The current nested customization argument value to be\n                modified.\n            conversion_fn: function. The function to be used for converting the\n                content. It is passed the customization argument value and\n                schema obj_type, which is one of 'SubtitledUnicode' or\n                'SubtitledHtml'.\n\n        Returns:\n            dict. The converted customization dict.\n        "
        is_subtitled_html_spec = schema['type'] == schema_utils.SCHEMA_TYPE_CUSTOM and schema['obj_type'] == schema_utils.SCHEMA_OBJ_TYPE_SUBTITLED_HTML
        is_subtitled_unicode_spec = schema['type'] == schema_utils.SCHEMA_TYPE_CUSTOM and schema['obj_type'] == schema_utils.SCHEMA_OBJ_TYPE_SUBTITLED_UNICODE
        if is_subtitled_html_spec or is_subtitled_unicode_spec:
            value = conversion_fn(value, schema['obj_type'])
        elif schema['type'] == schema_utils.SCHEMA_TYPE_LIST:
            assert isinstance(value, list)
            value = [InteractionCustomizationArg.traverse_by_schema_and_convert(schema['items'], value_element, conversion_fn) for value_element in value]
        elif schema['type'] == schema_utils.SCHEMA_TYPE_DICT:
            assert isinstance(value, dict)
            for property_spec in schema['properties']:
                name = property_spec['name']
                value[name] = InteractionCustomizationArg.traverse_by_schema_and_convert(property_spec['schema'], value[name], conversion_fn)
        return value

    @staticmethod
    def traverse_by_schema_and_get(schema: Dict[str, Any], value: Any, obj_types_to_search_for: List[str], value_extractor: Union[Callable[..., str], Callable[..., None]]) -> List[Any]:
        if False:
            while True:
                i = 10
        "Recursively traverses an interaction customization argument spec to\n        locate values with schema obj_type in obj_types_to_search_for, and\n        extracting the value using a value_extractor function.\n\n        Args:\n            schema: dict. The customization dict to be modified: dict\n                with a single key, 'value', whose corresponding value is the\n                value of the customization arg.\n            value: dict. The current nested customization argument value to be\n                modified.\n            obj_types_to_search_for: list(str). The obj types to search for. If\n                this list contains the current obj type, the value is passed to\n                value_extractor and the results are collected.\n            value_extractor: function. The function that extracts the wanted\n                computed value from each value that matches the obj_types. It\n                accepts one parameter, the value that matches the search object\n                type, and returns a desired computed value.\n\n        Returns:\n            list(*). A list of the extracted values returned from\n            value_extractor, which is run on any values that have a schema type\n            equal to 'custom' and have a obj_type in obj_types_to_search_for.\n            Because value_extractor can return any type, the result is a list of\n            any type.\n        "
        result = []
        schema_type = schema['type']
        if schema_type == schema_utils.SCHEMA_TYPE_CUSTOM and schema['obj_type'] in obj_types_to_search_for:
            result.append(value_extractor(value))
        elif schema_type == schema_utils.SCHEMA_TYPE_LIST:
            result = list(itertools.chain.from_iterable([InteractionCustomizationArg.traverse_by_schema_and_get(schema['items'], value_element, obj_types_to_search_for, value_extractor) for value_element in value]))
        elif schema_type == schema_utils.SCHEMA_TYPE_DICT:
            result = list(itertools.chain.from_iterable([InteractionCustomizationArg.traverse_by_schema_and_get(property_spec['schema'], value[property_spec['name']], obj_types_to_search_for, value_extractor) for property_spec in schema['properties']]))
        return result

    @staticmethod
    def convert_cust_args_dict_to_cust_args_based_on_specs(ca_dict: CustomizationArgsDictType, ca_specs_dict: List[domain.CustomizationArgSpecsDict]) -> Dict[str, InteractionCustomizationArg]:
        if False:
            for i in range(10):
                print('nop')
        "Converts customization arguments dictionary to customization\n        arguments. This is done by converting each customization argument to a\n        InteractionCustomizationArg domain object.\n\n        Args:\n            ca_dict: dict. A dictionary of customization\n                argument name to a customization argument dict, which is a dict\n                of the single key 'value' to the value of the customization\n                argument.\n            ca_specs_dict: dict. A dictionary of customization argument specs.\n\n        Returns:\n            dict. A dictionary of customization argument names to the\n            InteractionCustomizationArg domain object's.\n        "
        return {spec['name']: InteractionCustomizationArg.from_customization_arg_dict(ca_dict[spec['name']], spec['schema']) for spec in ca_specs_dict}

class OutcomeDict(TypedDict):
    """Dictionary representing the Outcome object."""
    dest: Optional[str]
    dest_if_really_stuck: Optional[str]
    feedback: SubtitledHtmlDict
    labelled_as_correct: bool
    param_changes: List[param_domain.ParamChangeDict]
    refresher_exploration_id: Optional[str]
    missing_prerequisite_skill_id: Optional[str]

class Outcome(translation_domain.BaseTranslatableObject):
    """Value object representing an outcome of an interaction. An outcome
    consists of a destination state, feedback to show the user, and any
    parameter changes.
    """

    def __init__(self, dest: Optional[str], dest_if_really_stuck: Optional[str], feedback: SubtitledHtml, labelled_as_correct: bool, param_changes: List[param_domain.ParamChange], refresher_exploration_id: Optional[str], missing_prerequisite_skill_id: Optional[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initializes a Outcome domain object.\n\n        Args:\n            dest: str. The name of the destination state.\n            dest_if_really_stuck: str or None. The name of the optional state\n                to redirect the learner to strengthen their concepts.\n            feedback: SubtitledHtml. Feedback to give to the user if this rule\n                is triggered.\n            labelled_as_correct: bool. Whether this outcome has been labelled\n                by the creator as corresponding to a "correct" answer.\n            param_changes: list(ParamChange). List of exploration-level\n                parameter changes to make if this rule is triggered.\n            refresher_exploration_id: str or None. An optional exploration ID\n                to redirect the learner to if they seem to lack understanding\n                of a prerequisite concept. This should only exist if the\n                destination state for this outcome is a self-loop.\n            missing_prerequisite_skill_id: str or None. The id of the skill that\n                this answer group tests. If this is not None, the exploration\n                player would redirect to this skill when a learner receives this\n                outcome.\n        '
        self.dest = dest
        self.dest_if_really_stuck = dest_if_really_stuck
        self.feedback = feedback
        self.labelled_as_correct = labelled_as_correct
        self.param_changes = param_changes or []
        self.refresher_exploration_id = refresher_exploration_id
        self.missing_prerequisite_skill_id = missing_prerequisite_skill_id

    def get_translatable_contents_collection(self, **kwargs: Optional[str]) -> translation_domain.TranslatableContentsCollection:
        if False:
            return 10
        'Get all translatable fields in the outcome.\n\n        Returns:\n            translatable_contents_collection: TranslatableContentsCollection.\n            An instance of TranslatableContentsCollection class.\n        '
        translatable_contents_collection = translation_domain.TranslatableContentsCollection()
        translatable_contents_collection.add_translatable_field(self.feedback.content_id, translation_domain.ContentType.FEEDBACK, translation_domain.TranslatableContentFormat.HTML, self.feedback.html)
        return translatable_contents_collection

    def to_dict(self) -> OutcomeDict:
        if False:
            print('Hello World!')
        'Returns a dict representing this Outcome domain object.\n\n        Returns:\n            dict. A dict, mapping all fields of Outcome instance.\n        '
        return {'dest': self.dest, 'dest_if_really_stuck': self.dest_if_really_stuck, 'feedback': self.feedback.to_dict(), 'labelled_as_correct': self.labelled_as_correct, 'param_changes': [param_change.to_dict() for param_change in self.param_changes], 'refresher_exploration_id': self.refresher_exploration_id, 'missing_prerequisite_skill_id': self.missing_prerequisite_skill_id}

    @classmethod
    def from_dict(cls, outcome_dict: OutcomeDict, validate: bool=True) -> Outcome:
        if False:
            i = 10
            return i + 15
        'Return a Outcome domain object from a dict.\n\n        Args:\n            outcome_dict: dict. The dict representation of Outcome object.\n            validate: bool. False, when the validations should not be called.\n\n        Returns:\n            Outcome. The corresponding Outcome domain object.\n        '
        feedback = SubtitledHtml.from_dict(outcome_dict['feedback'])
        if validate:
            feedback.validate()
        return cls(outcome_dict['dest'], outcome_dict['dest_if_really_stuck'], feedback, outcome_dict['labelled_as_correct'], [param_domain.ParamChange(param_change['name'], param_change['generator_id'], param_change['customization_args']) for param_change in outcome_dict['param_changes']], outcome_dict['refresher_exploration_id'], outcome_dict['missing_prerequisite_skill_id'])

    def validate(self) -> None:
        if False:
            return 10
        'Validates various properties of the Outcome.\n\n        Raises:\n            ValidationError. One or more attributes of the Outcome are invalid.\n        '
        self.feedback.validate()
        if not isinstance(self.labelled_as_correct, bool):
            raise utils.ValidationError('The "labelled_as_correct" field should be a boolean, received %s' % self.labelled_as_correct)
        if self.missing_prerequisite_skill_id is not None:
            if not isinstance(self.missing_prerequisite_skill_id, str):
                raise utils.ValidationError('Expected outcome missing_prerequisite_skill_id to be a string, received %s' % self.missing_prerequisite_skill_id)
        if not isinstance(self.param_changes, list):
            raise utils.ValidationError('Expected outcome param_changes to be a list, received %s' % self.param_changes)
        for param_change in self.param_changes:
            param_change.validate()
        if self.refresher_exploration_id is not None:
            if not isinstance(self.refresher_exploration_id, str):
                raise utils.ValidationError('Expected outcome refresher_exploration_id to be a string, received %s' % self.refresher_exploration_id)

    @staticmethod
    def convert_html_in_outcome(outcome_dict: OutcomeDict, conversion_fn: Callable[[str], str]) -> OutcomeDict:
        if False:
            print('Hello World!')
        'Checks for HTML fields in the outcome and converts it\n        according to the conversion function.\n\n        Args:\n            outcome_dict: dict. The outcome dict.\n            conversion_fn: function. The function to be used for converting the\n                HTML.\n\n        Returns:\n            dict. The converted outcome dict.\n        '
        outcome_dict['feedback']['html'] = conversion_fn(outcome_dict['feedback']['html'])
        return outcome_dict

class VoiceoverDict(TypedDict):
    """Dictionary representing the Voiceover object."""
    filename: str
    file_size_bytes: int
    needs_update: bool
    duration_secs: float

class Voiceover:
    """Value object representing an voiceover."""

    def to_dict(self) -> VoiceoverDict:
        if False:
            return 10
        'Returns a dict representing this Voiceover domain object.\n\n        Returns:\n            dict. A dict, mapping all fields of Voiceover instance.\n        '
        return {'filename': self.filename, 'file_size_bytes': self.file_size_bytes, 'needs_update': self.needs_update, 'duration_secs': self.duration_secs}

    @classmethod
    def from_dict(cls, voiceover_dict: VoiceoverDict) -> Voiceover:
        if False:
            while True:
                i = 10
        'Return a Voiceover domain object from a dict.\n\n        Args:\n            voiceover_dict: dict. The dict representation of\n                Voiceover object.\n\n        Returns:\n            Voiceover. The corresponding Voiceover domain object.\n        '
        return cls(voiceover_dict['filename'], voiceover_dict['file_size_bytes'], voiceover_dict['needs_update'], voiceover_dict['duration_secs'])

    def __init__(self, filename: str, file_size_bytes: int, needs_update: bool, duration_secs: float) -> None:
        if False:
            i = 10
            return i + 15
        'Initializes a Voiceover domain object.\n\n        Args:\n            filename: str. The corresponding voiceover file path.\n            file_size_bytes: int. The file size, in bytes. Used to display\n                potential bandwidth usage to the learner before they download\n                the file.\n            needs_update: bool. Whether voiceover is marked for needing review.\n            duration_secs: float. The duration in seconds for the voiceover\n                recording.\n        '
        self.filename = filename
        self.file_size_bytes = file_size_bytes
        self.needs_update = needs_update
        self.duration_secs = duration_secs

    def validate(self) -> None:
        if False:
            return 10
        'Validates properties of the Voiceover.\n\n        Raises:\n            ValidationError. One or more attributes of the Voiceover are\n                invalid.\n        '
        if not isinstance(self.filename, str):
            raise utils.ValidationError('Expected audio filename to be a string, received %s' % self.filename)
        dot_index = self.filename.rfind('.')
        if dot_index in (-1, 0):
            raise utils.ValidationError('Invalid audio filename: %s' % self.filename)
        extension = self.filename[dot_index + 1:]
        if extension not in feconf.ACCEPTED_AUDIO_EXTENSIONS:
            raise utils.ValidationError('Invalid audio filename: it should have one of the following extensions: %s. Received: %s' % (list(feconf.ACCEPTED_AUDIO_EXTENSIONS.keys()), self.filename))
        if not isinstance(self.file_size_bytes, int):
            raise utils.ValidationError('Expected file size to be an int, received %s' % self.file_size_bytes)
        if self.file_size_bytes <= 0:
            raise utils.ValidationError('Invalid file size: %s' % self.file_size_bytes)
        if not isinstance(self.needs_update, bool):
            raise utils.ValidationError('Expected needs_update to be a bool, received %s' % self.needs_update)
        if not isinstance(self.duration_secs, (float, int)):
            raise utils.ValidationError('Expected duration_secs to be a float, received %s' % self.duration_secs)
        if self.duration_secs < 0:
            raise utils.ValidationError('Expected duration_secs to be positive number, or zero if not yet specified %s' % self.duration_secs)

class RecordedVoiceoversDict(TypedDict):
    """Dictionary representing the RecordedVoiceovers object."""
    voiceovers_mapping: Dict[str, Dict[str, VoiceoverDict]]

class RecordedVoiceovers:
    """Value object representing a recorded voiceovers which stores voiceover of
    all state contents (like hints, feedback etc.) in different languages linked
    through their content_id.
    """

    def __init__(self, voiceovers_mapping: Dict[str, Dict[str, Voiceover]]) -> None:
        if False:
            i = 10
            return i + 15
        'Initializes a RecordedVoiceovers domain object.\n\n        Args:\n            voiceovers_mapping: dict. A dict mapping the content Ids\n                to the dicts which is the map of abbreviated code of the\n                languages to the Voiceover objects.\n        '
        self.voiceovers_mapping = voiceovers_mapping

    def to_dict(self) -> RecordedVoiceoversDict:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict representing this RecordedVoiceovers domain object.\n\n        Returns:\n            dict. A dict, mapping all fields of RecordedVoiceovers instance.\n        '
        voiceovers_mapping: Dict[str, Dict[str, VoiceoverDict]] = {}
        for (content_id, language_code_to_voiceover) in self.voiceovers_mapping.items():
            voiceovers_mapping[content_id] = {}
            for (language_code, voiceover) in language_code_to_voiceover.items():
                voiceovers_mapping[content_id][language_code] = voiceover.to_dict()
        recorded_voiceovers_dict: RecordedVoiceoversDict = {'voiceovers_mapping': voiceovers_mapping}
        return recorded_voiceovers_dict

    @classmethod
    def from_dict(cls, recorded_voiceovers_dict: RecordedVoiceoversDict) -> RecordedVoiceovers:
        if False:
            return 10
        'Return a RecordedVoiceovers domain object from a dict.\n\n        Args:\n            recorded_voiceovers_dict: dict. The dict representation of\n                RecordedVoiceovers object.\n\n        Returns:\n            RecordedVoiceovers. The corresponding RecordedVoiceovers domain\n            object.\n        '
        voiceovers_mapping: Dict[str, Dict[str, Voiceover]] = {}
        for (content_id, language_code_to_voiceover) in recorded_voiceovers_dict['voiceovers_mapping'].items():
            voiceovers_mapping[content_id] = {}
            for (language_code, voiceover) in language_code_to_voiceover.items():
                voiceovers_mapping[content_id][language_code] = Voiceover.from_dict(voiceover)
        return cls(voiceovers_mapping)

    def validate(self, expected_content_id_list: Optional[List[str]]) -> None:
        if False:
            print('Hello World!')
        'Validates properties of the RecordedVoiceovers.\n\n        Args:\n            expected_content_id_list: list(str)|None. A list of content id which\n                are expected to be inside the RecordedVoiceovers.\n\n        Raises:\n            ValidationError. One or more attributes of the RecordedVoiceovers\n                are invalid.\n        '
        if expected_content_id_list is not None:
            if not set(self.voiceovers_mapping.keys()) == set(expected_content_id_list):
                raise utils.ValidationError('Expected state recorded_voiceovers to match the listed content ids %s, found %s' % (expected_content_id_list, list(self.voiceovers_mapping.keys())))
        for (content_id, language_code_to_voiceover) in self.voiceovers_mapping.items():
            if not isinstance(content_id, str):
                raise utils.ValidationError('Expected content_id to be a string, received %s' % content_id)
            if not isinstance(language_code_to_voiceover, dict):
                raise utils.ValidationError('Expected content_id value to be a dict, received %s' % language_code_to_voiceover)
            for (language_code, voiceover) in language_code_to_voiceover.items():
                if not isinstance(language_code, str):
                    raise utils.ValidationError('Expected language_code to be a string, received %s' % language_code)
                allowed_language_codes = [language['id'] for language in constants.SUPPORTED_AUDIO_LANGUAGES]
                if language_code not in allowed_language_codes:
                    raise utils.ValidationError('Invalid language_code: %s' % language_code)
                voiceover.validate()

    def get_content_ids_for_voiceovers(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        'Returns a list of content_id available for voiceover.\n\n        Returns:\n            list(str). A list of content id available for voiceover.\n        '
        return list(self.voiceovers_mapping.keys())

    def strip_all_existing_voiceovers(self) -> None:
        if False:
            print('Hello World!')
        'Strips all existing voiceovers from the voiceovers_mapping.'
        for content_id in self.voiceovers_mapping.keys():
            self.voiceovers_mapping[content_id] = {}

    def add_content_id_for_voiceover(self, content_id: str) -> None:
        if False:
            print('Hello World!')
        "Adds a content id as a key for the voiceover into the\n        voiceovers_mapping dict.\n\n        Args:\n            content_id: str. The id representing a subtitled html.\n\n        Raises:\n            Exception. The content id isn't a string.\n            Exception. The content id already exist in the voiceovers_mapping\n                dict.\n        "
        if not isinstance(content_id, str):
            raise Exception('Expected content_id to be a string, received %s' % content_id)
        if content_id in self.voiceovers_mapping:
            raise Exception('The content_id %s already exist.' % content_id)
        self.voiceovers_mapping[content_id] = {}

    def delete_content_id_for_voiceover(self, content_id: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Deletes a content id from the voiceovers_mapping dict.\n\n        Args:\n            content_id: str. The id representing a subtitled html.\n\n        Raises:\n            Exception. The content id isn't a string.\n            Exception. The content id does not exist in the voiceovers_mapping\n                dict.\n        "
        if not isinstance(content_id, str):
            raise Exception('Expected content_id to be a string, received %s' % content_id)
        if content_id not in self.voiceovers_mapping:
            raise Exception('The content_id %s does not exist.' % content_id)
        self.voiceovers_mapping.pop(content_id, None)

class RuleSpecDict(TypedDict):
    """Dictionary representing the RuleSpec object."""
    rule_type: str
    inputs: Dict[str, AllowedRuleSpecInputTypes]

class RuleSpec(translation_domain.BaseTranslatableObject):
    """Value object representing a rule specification."""

    def __init__(self, rule_type: str, inputs: Mapping[str, AllowedRuleSpecInputTypes]) -> None:
        if False:
            print('Hello World!')
        'Initializes a RuleSpec domain object.\n\n        Args:\n            rule_type: str. The rule type, e.g. "CodeContains" or "Equals". A\n                full list of rule types can be found in\n                extensions/interactions/rule_templates.json.\n            inputs: dict. The values of the parameters needed in order to fully\n                specify the rule. The keys for this dict can be deduced from\n                the relevant description field in\n                extensions/interactions/rule_templates.json -- they are\n                enclosed in {{...}} braces.\n        '
        self.rule_type = rule_type
        assert isinstance(inputs, dict)
        self.inputs = inputs

    def get_translatable_contents_collection(self, **kwargs: Optional[str]) -> translation_domain.TranslatableContentsCollection:
        if False:
            return 10
        'Get all translatable fields in the rule spec.\n\n        Returns:\n            translatable_contents_collection: TranslatableContentsCollection.\n            An instance of TranslatableContentsCollection class.\n        '
        translatable_contents_collection = translation_domain.TranslatableContentsCollection()
        for input_value in self.inputs.values():
            if 'normalizedStrSet' in input_value:
                translatable_contents_collection.add_translatable_field(input_value['contentId'], translation_domain.ContentType.RULE, translation_domain.TranslatableContentFormat.SET_OF_NORMALIZED_STRING, input_value['normalizedStrSet'], kwargs['interaction_id'], self.rule_type)
            if 'unicodeStrSet' in input_value:
                translatable_contents_collection.add_translatable_field(input_value['contentId'], translation_domain.ContentType.RULE, translation_domain.TranslatableContentFormat.SET_OF_UNICODE_STRING, input_value['unicodeStrSet'], kwargs['interaction_id'], self.rule_type)
        return translatable_contents_collection

    def to_dict(self) -> RuleSpecDict:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict representing this RuleSpec domain object.\n\n        Returns:\n            dict. A dict, mapping all fields of RuleSpec instance.\n        '
        return {'rule_type': self.rule_type, 'inputs': self.inputs}

    @classmethod
    def from_dict(cls, rulespec_dict: RuleSpecDict) -> RuleSpec:
        if False:
            for i in range(10):
                print('nop')
        'Return a RuleSpec domain object from a dict.\n\n        Args:\n            rulespec_dict: dict. The dict representation of RuleSpec object.\n\n        Returns:\n            RuleSpec. The corresponding RuleSpec domain object.\n        '
        return cls(rulespec_dict['rule_type'], rulespec_dict['inputs'])

    def validate(self, rule_params_list: List[Tuple[str, Type[objects.BaseObject]]], exp_param_specs_dict: Dict[str, param_domain.ParamSpec]) -> None:
        if False:
            while True:
                i = 10
        'Validates a RuleSpec value object. It ensures the inputs dict does\n        not refer to any non-existent parameters and that it contains values\n        for all the parameters the rule expects.\n\n        Args:\n            rule_params_list: list(str, object(*)). A list of parameters used by\n                the rule represented by this RuleSpec instance, to be used to\n                validate the inputs of this RuleSpec. Each element of the list\n                represents a single parameter and is a tuple with two elements:\n                    0: The name (string) of the parameter.\n                    1: The typed object instance for that\n                        parameter (e.g. Real).\n            exp_param_specs_dict: dict. A dict of specified parameters used in\n                this exploration. Keys are parameter names and values are\n                ParamSpec value objects with an object type property (obj_type).\n                RuleSpec inputs may have a parameter value which refers to one\n                of these exploration parameters.\n\n        Raises:\n            ValidationError. One or more attributes of the RuleSpec are\n                invalid.\n        '
        if not isinstance(self.inputs, dict):
            raise utils.ValidationError('Expected inputs to be a dict, received %s' % self.inputs)
        input_key_set = set(self.inputs.keys())
        param_names_set = set((rp[0] for rp in rule_params_list))
        leftover_input_keys = input_key_set - param_names_set
        leftover_param_names = param_names_set - input_key_set
        if leftover_input_keys:
            logging.warning("RuleSpec '%s' has inputs which are not recognized parameter names: %s" % (self.rule_type, leftover_input_keys))
        if leftover_param_names:
            raise utils.ValidationError("RuleSpec '%s' is missing inputs: %s" % (self.rule_type, leftover_param_names))
        rule_params_dict = {rp[0]: rp[1] for rp in rule_params_list}
        for (param_name, param_value) in self.inputs.items():
            param_obj = rule_params_dict[param_name]
            if isinstance(param_value, str) and '{{' in param_value:
                start_brace_index = param_value.index('{{') + 2
                end_brace_index = param_value.index('}}')
                param_spec_name = param_value[start_brace_index:end_brace_index]
                if param_spec_name not in exp_param_specs_dict:
                    raise utils.ValidationError("RuleSpec '%s' has an input with name '%s' which refers to an unknown parameter within the exploration: %s" % (self.rule_type, param_name, param_spec_name))
            else:
                param_obj.normalize(param_value)

    @staticmethod
    def convert_html_in_rule_spec(rule_spec_dict: RuleSpecDict, conversion_fn: Callable[[str], str], html_field_types_to_rule_specs: Dict[str, rules_registry.RuleSpecsExtensionDict]) -> RuleSpecDict:
        if False:
            print('Hello World!')
        'Checks for HTML fields in a Rule Spec and converts it according\n        to the conversion function.\n\n        Args:\n            rule_spec_dict: dict. The Rule Spec dict.\n            conversion_fn: function. The function to be used for converting the\n                HTML.\n            html_field_types_to_rule_specs: dict. A dictionary that specifies\n                the locations of html fields in rule specs. It is defined as a\n                mapping of rule input types to a dictionary containing\n                interaction id, format, and rule types. See\n                html_field_types_to_rule_specs_state_v41.json for an example.\n\n        Returns:\n            dict. The converted Rule Spec dict.\n\n        Raises:\n            Exception. The Rule spec has an invalid format.\n            Exception. The Rule spec has no valid input variable\n                with HTML in it.\n        '
        for interaction_and_rule_details in html_field_types_to_rule_specs.values():
            rule_type_has_html = rule_spec_dict['rule_type'] in interaction_and_rule_details['ruleTypes'].keys()
            if rule_type_has_html:
                html_type_format = interaction_and_rule_details['format']
                input_variables_from_html_mapping = interaction_and_rule_details['ruleTypes'][rule_spec_dict['rule_type']]['htmlInputVariables']
                input_variable_match_found = False
                for input_variable in rule_spec_dict['inputs'].keys():
                    if input_variable in input_variables_from_html_mapping:
                        input_variable_match_found = True
                        rule_input_variable = rule_spec_dict['inputs'][input_variable]
                        if html_type_format == feconf.HTML_RULE_VARIABLE_FORMAT_STRING:
                            input_value = rule_spec_dict['inputs'][input_variable]
                            assert isinstance(input_value, str)
                            rule_spec_dict['inputs'][input_variable] = conversion_fn(input_value)
                        elif html_type_format == feconf.HTML_RULE_VARIABLE_FORMAT_SET:
                            if isinstance(rule_input_variable, list):
                                for (value_index, value) in enumerate(rule_input_variable):
                                    if isinstance(value, str):
                                        variable_format_set_input = cast(Dict[str, List[str]], rule_spec_dict['inputs'])
                                        variable_format_set_input[input_variable][value_index] = conversion_fn(value)
                        elif html_type_format == feconf.HTML_RULE_VARIABLE_FORMAT_LIST_OF_SETS:
                            input_variable_list = rule_spec_dict['inputs'][input_variable]
                            assert isinstance(input_variable_list, list)
                            for (list_index, html_list) in enumerate(input_variable_list):
                                for (rule_html_index, rule_html) in enumerate(html_list):
                                    list_of_sets_inputs = cast(Dict[str, List[List[str]]], rule_spec_dict['inputs'])
                                    list_of_sets_inputs[input_variable][list_index][rule_html_index] = conversion_fn(rule_html)
                        else:
                            raise Exception('The rule spec does not belong to a valid format.')
                if not input_variable_match_found:
                    raise Exception('Rule spec should have at least one valid input variable with Html in it.')
        return rule_spec_dict

class SubtitledHtmlDict(TypedDict):
    """Dictionary representing the SubtitledHtml object."""
    content_id: str
    html: str

class SubtitledHtml:
    """Value object representing subtitled HTML."""

    def __init__(self, content_id: str, html: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initializes a SubtitledHtml domain object. Note that initializing\n        the SubtitledHtml object does not clean the html. This is because we\n        sometimes need to initialize SubtitledHtml and migrate the contained\n        html from an old schema, but the cleaner would remove invalid tags\n        and attributes before having a chance to migrate it. An example where\n        this functionality is required is\n        InteractionInstance.convert_html_in_interaction. Before saving the\n        SubtitledHtml object, validate() should be called for validation and\n        cleaning of the html.\n\n        Args:\n            content_id: str. A unique id referring to the other assets for this\n                content.\n            html: str. A piece of user-submitted HTML. Note that this is NOT\n                cleaned in such a way as to contain a restricted set of HTML\n                tags. To clean it, the validate() method must be called.\n        '
        self.content_id = content_id
        self.html = html

    def to_dict(self) -> SubtitledHtmlDict:
        if False:
            i = 10
            return i + 15
        'Returns a dict representing this SubtitledHtml domain object.\n\n        Returns:\n            dict. A dict, mapping all fields of SubtitledHtml instance.\n        '
        return {'content_id': self.content_id, 'html': self.html}

    @classmethod
    def from_dict(cls, subtitled_html_dict: SubtitledHtmlDict) -> SubtitledHtml:
        if False:
            while True:
                i = 10
        'Return a SubtitledHtml domain object from a dict.\n\n        Args:\n            subtitled_html_dict: dict. The dict representation of SubtitledHtml\n                object.\n\n        Returns:\n            SubtitledHtml. The corresponding SubtitledHtml domain object.\n        '
        return cls(subtitled_html_dict['content_id'], subtitled_html_dict['html'])

    def validate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Validates properties of the SubtitledHtml, and cleans the html.\n\n        Raises:\n            ValidationError. One or more attributes of the SubtitledHtml are\n                invalid.\n        '
        if not isinstance(self.content_id, str):
            raise utils.ValidationError('Expected content id to be a string, received %s' % self.content_id)
        if not isinstance(self.html, str):
            raise utils.ValidationError('Invalid content HTML: %s' % self.html)
        self.html = html_cleaner.clean(self.html)
        html_cleaner.validate_rte_tags(self.html)
        html_cleaner.validate_tabs_and_collapsible_rte_tags(self.html)

    @classmethod
    def create_default_subtitled_html(cls, content_id: str) -> SubtitledHtml:
        if False:
            for i in range(10):
                print('nop')
        "Create a default SubtitledHtml domain object.\n\n        Args:\n            content_id: str. The id of the content.\n\n        Returns:\n            SubtitledHtml. A default SubtitledHtml domain object, some\n            attribute of that object will be ''.\n        "
        return cls(content_id, '')

class SubtitledUnicodeDict(TypedDict):
    """Dictionary representing the SubtitledUnicode object."""
    content_id: str
    unicode_str: str

class SubtitledUnicode:
    """Value object representing subtitled unicode."""

    def __init__(self, content_id: str, unicode_str: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initializes a SubtitledUnicode domain object.\n\n        Args:\n            content_id: str. A unique id referring to the other assets for this\n                content.\n            unicode_str: str. A piece of user-submitted unicode.\n        '
        self.content_id = content_id
        self.unicode_str = unicode_str
        self.validate()

    def to_dict(self) -> SubtitledUnicodeDict:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict representing this SubtitledUnicode domain object.\n\n        Returns:\n            dict. A dict, mapping all fields of SubtitledUnicode instance.\n        '
        return {'content_id': self.content_id, 'unicode_str': self.unicode_str}

    @classmethod
    def from_dict(cls, subtitled_unicode_dict: SubtitledUnicodeDict) -> SubtitledUnicode:
        if False:
            return 10
        'Return a SubtitledUnicode domain object from a dict.\n\n        Args:\n            subtitled_unicode_dict: dict. The dict representation of\n                SubtitledUnicode object.\n\n        Returns:\n            SubtitledUnicode. The corresponding SubtitledUnicode domain object.\n        '
        return cls(subtitled_unicode_dict['content_id'], subtitled_unicode_dict['unicode_str'])

    def validate(self) -> None:
        if False:
            print('Hello World!')
        'Validates properties of the SubtitledUnicode.\n\n        Raises:\n            ValidationError. One or more attributes of the SubtitledUnicode are\n                invalid.\n        '
        if not isinstance(self.content_id, str):
            raise utils.ValidationError('Expected content id to be a string, received %s' % self.content_id)
        if not isinstance(self.unicode_str, str):
            raise utils.ValidationError('Invalid content unicode: %s' % self.unicode_str)

    @classmethod
    def create_default_subtitled_unicode(cls, content_id: str) -> SubtitledUnicode:
        if False:
            print('Hello World!')
        'Create a default SubtitledUnicode domain object.\n\n        Args:\n            content_id: str. The id of the content.\n\n        Returns:\n            SubtitledUnicode. A default SubtitledUnicode domain object.\n        '
        return cls(content_id, '')
DomainObjectCustomizationArgsConversionFnTypes = Union[Callable[[SubtitledHtml, str], SubtitledHtml], Callable[[SubtitledHtml, str], SubtitledHtmlDict], Callable[[SubtitledUnicode, str], SubtitledUnicodeDict], Callable[[SubtitledHtml, str], List[str]]]
DictCustomizationArgsConversionFnTypes = Union[Callable[[Dict[str, str], Literal['SubtitledUnicode']], SubtitledUnicode], Callable[[Dict[str, str], Literal['SubtitledHtml']], SubtitledHtml]]
AcceptableConversionFnType = Union[DomainObjectCustomizationArgsConversionFnTypes, DictCustomizationArgsConversionFnTypes]

class StateDict(TypedDict):
    """Dictionary representing the State object."""
    content: SubtitledHtmlDict
    param_changes: List[param_domain.ParamChangeDict]
    interaction: InteractionInstanceDict
    recorded_voiceovers: RecordedVoiceoversDict
    solicit_answer_details: bool
    card_is_checkpoint: bool
    linked_skill_id: Optional[str]
    classifier_model_id: Optional[str]

class State(translation_domain.BaseTranslatableObject):
    """Domain object for a state."""

    def __init__(self, content: SubtitledHtml, param_changes: List[param_domain.ParamChange], interaction: InteractionInstance, recorded_voiceovers: RecordedVoiceovers, solicit_answer_details: bool, card_is_checkpoint: bool, linked_skill_id: Optional[str]=None, classifier_model_id: Optional[str]=None) -> None:
        if False:
            print('Hello World!')
        'Initializes a State domain object.\n\n        Args:\n            content: SubtitledHtml. The contents displayed to the reader in this\n                state.\n            param_changes: list(ParamChange). Parameter changes associated with\n                this state.\n            interaction: InteractionInstance. The interaction instance\n                associated with this state.\n            recorded_voiceovers: RecordedVoiceovers. The recorded voiceovers for\n                the state contents and translations.\n            solicit_answer_details: bool. Whether the creator wants to ask\n                for answer details from the learner about why they picked a\n                particular answer while playing the exploration.\n            card_is_checkpoint: bool. If the card is marked as a checkpoint by\n                the creator or not.\n            linked_skill_id: str or None. The linked skill ID associated with\n                this state.\n            classifier_model_id: str or None. The classifier model ID\n                associated with this state, if applicable.\n        '
        self.content = content
        self.param_changes = [param_domain.ParamChange(param_change.name, param_change.generator.id, param_change.customization_args) for param_change in param_changes]
        self.interaction = InteractionInstance(interaction.id, interaction.customization_args, interaction.answer_groups, interaction.default_outcome, interaction.confirmed_unclassified_answers, interaction.hints, interaction.solution)
        self.classifier_model_id = classifier_model_id
        self.recorded_voiceovers = recorded_voiceovers
        self.linked_skill_id = linked_skill_id
        self.solicit_answer_details = solicit_answer_details
        self.card_is_checkpoint = card_is_checkpoint

    def get_translatable_contents_collection(self, **kwargs: Optional[str]) -> translation_domain.TranslatableContentsCollection:
        if False:
            return 10
        'Get all translatable fields in the state.\n\n        Returns:\n            translatable_contents_collection: TranslatableContentsCollection.\n            An instance of TranslatableContentsCollection class.\n        '
        translatable_contents_collection = translation_domain.TranslatableContentsCollection()
        translatable_contents_collection.add_translatable_field(self.content.content_id, translation_domain.ContentType.CONTENT, translation_domain.TranslatableContentFormat.HTML, self.content.html)
        translatable_contents_collection.add_fields_from_translatable_object(self.interaction)
        return translatable_contents_collection

    def validate(self, exp_param_specs_dict: Optional[Dict[str, param_domain.ParamSpec]], allow_null_interaction: bool, *, tagged_skill_misconception_id_required: bool=False, strict: bool=False) -> None:
        if False:
            return 10
        "Validates various properties of the State.\n\n        Args:\n            exp_param_specs_dict: dict or None. A dict of specified parameters\n                used in this exploration. Keys are parameter names and values\n                are ParamSpec value objects with an object type\n                property(obj_type). It is None if the state belongs to a\n                question.\n            allow_null_interaction: bool. Whether this state's interaction is\n                allowed to be unspecified.\n            tagged_skill_misconception_id_required: bool. The 'tagged_skill_\n                misconception_id' is required or not.\n            strict: bool. Tells if the validation is strict or not. Validation\n                should be strict for all published entities, i.e. those that\n                are viewable by a learner. It can be non-strict for entities\n                that are only viewable by lesson creators.\n\n        Raises:\n            ValidationError. One or more attributes of the State are invalid.\n        "
        self.content.validate()
        if exp_param_specs_dict:
            param_specs_dict = exp_param_specs_dict
        else:
            param_specs_dict = {}
        if not isinstance(self.param_changes, list):
            raise utils.ValidationError('Expected state param_changes to be a list, received %s' % self.param_changes)
        for param_change in self.param_changes:
            param_change.validate()
        if not allow_null_interaction and self.interaction.id is None:
            raise utils.ValidationError('This state does not have any interaction specified.')
        if self.interaction.id is not None:
            self.interaction.validate(param_specs_dict, tagged_skill_misconception_id_required=tagged_skill_misconception_id_required, strict=strict)
        if not isinstance(self.solicit_answer_details, bool):
            raise utils.ValidationError('Expected solicit_answer_details to be a boolean, received %s' % self.solicit_answer_details)
        if self.solicit_answer_details:
            if self.interaction.id in constants.INTERACTION_IDS_WITHOUT_ANSWER_DETAILS:
                raise utils.ValidationError('The %s interaction does not support soliciting answer details from learners.' % self.interaction.id)
        if not isinstance(self.card_is_checkpoint, bool):
            raise utils.ValidationError('Expected card_is_checkpoint to be a boolean, received %s' % self.card_is_checkpoint)
        self.recorded_voiceovers.validate(self.get_translatable_content_ids())
        if self.linked_skill_id is not None:
            if not isinstance(self.linked_skill_id, str):
                raise utils.ValidationError('Expected linked_skill_id to be a str, received %s.' % self.linked_skill_id)

    def is_rte_content_supported_on_android(self) -> bool:
        if False:
            print('Hello World!')
        'Checks whether the RTE components used in the state are supported by\n        Android.\n\n        Returns:\n            bool. Whether the RTE components in the state is valid.\n        '

        def require_valid_component_names(html: str) -> bool:
            if False:
                print('Hello World!')
            'Checks if the provided html string contains only allowed\n            RTE tags.\n\n            Args:\n                html: str. The html string.\n\n            Returns:\n                bool. Whether all RTE tags in the html are allowed.\n            '
            component_name_prefix = 'oppia-noninteractive-'
            component_names = set((component['id'].replace(component_name_prefix, '') for component in html_cleaner.get_rte_components(html)))
            return any(component_names.difference(android_validation_constants.VALID_RTE_COMPONENTS))
        if self.content and require_valid_component_names(self.content.html):
            return False
        return self.interaction.is_rte_content_supported_on_android(require_valid_component_names)

    def get_training_data(self) -> List[TrainingDataDict]:
        if False:
            print('Hello World!')
        "Retrieves training data from the State domain object.\n\n        Returns:\n            list(dict). A list of dicts, each of which has two key-value pairs.\n            One pair maps 'answer_group_index' to the index of the answer\n            group and the other maps 'answers' to the answer group's\n            training data.\n        "
        state_training_data_by_answer_group: List[TrainingDataDict] = []
        for (answer_group_index, answer_group) in enumerate(self.interaction.answer_groups):
            if answer_group.training_data:
                answers = copy.deepcopy(answer_group.training_data)
                state_training_data_by_answer_group.append({'answer_group_index': answer_group_index, 'answers': answers})
        return state_training_data_by_answer_group

    def can_undergo_classification(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Checks whether the answers for this state satisfy the preconditions\n        for a ML model to be trained.\n\n        Returns:\n            bool. True, if the conditions are satisfied.\n        '
        training_examples_count = 0
        labels_count = 0
        training_examples_count += len(self.interaction.confirmed_unclassified_answers)
        for answer_group in self.interaction.answer_groups:
            training_examples_count += len(answer_group.training_data)
            labels_count += 1
        if training_examples_count >= feconf.MIN_TOTAL_TRAINING_EXAMPLES and labels_count >= feconf.MIN_ASSIGNED_LABELS:
            return True
        return False

    @classmethod
    def convert_state_dict_to_yaml(cls, state_dict: StateDict, width: int) -> str:
        if False:
            while True:
                i = 10
        'Converts the given state dict to yaml format.\n\n        Args:\n            state_dict: dict. A dict representing a state in an exploration.\n            width: int. The maximum number of characters in a line for the\n                returned YAML string.\n\n        Returns:\n            str. The YAML version of the state_dict.\n\n        Raises:\n            Exception. The state dict does not represent a valid state.\n        '
        try:
            state = cls.from_dict(state_dict)
        except Exception as e:
            logging.exception('Bad state dict: %s' % str(state_dict))
            raise e
        return utils.yaml_from_dict(state.to_dict(), width=width)

    def _update_content_ids_in_assets(self, old_ids_list: List[str], new_ids_list: List[str]) -> None:
        if False:
            return 10
        "Adds or deletes content ids in assets i.e, other parts of state\n        object such as recorded_voiceovers.\n\n        Args:\n            old_ids_list: list(str). A list of content ids present earlier\n                within the substructure (like answer groups, hints etc.) of\n                state.\n            new_ids_list: list(str). A list of content ids currently present\n                within the substructure (like answer groups, hints etc.) of\n                state.\n\n        Raises:\n            Exception. The content to be deleted doesn't exist.\n            Exception. The content to be added already exists.\n        "
        content_ids_to_delete = set(old_ids_list) - set(new_ids_list)
        content_ids_to_add = set(new_ids_list) - set(old_ids_list)
        content_ids_for_voiceovers = self.recorded_voiceovers.get_content_ids_for_voiceovers()
        for content_id in content_ids_to_delete:
            if not content_id in content_ids_for_voiceovers:
                raise Exception('The content_id %s does not exist in recorded_voiceovers.' % content_id)
            self.recorded_voiceovers.delete_content_id_for_voiceover(content_id)
        for content_id in content_ids_to_add:
            if content_id in content_ids_for_voiceovers:
                raise Exception('The content_id %s already exists in recorded_voiceovers' % content_id)
            self.recorded_voiceovers.add_content_id_for_voiceover(content_id)

    def update_content(self, content: SubtitledHtml) -> None:
        if False:
            print('Hello World!')
        'Update the content of this state.\n\n        Args:\n            content: SubtitledHtml. Representation of updated content.\n        '
        old_content_id = self.content.content_id
        self.content = content
        self._update_content_ids_in_assets([old_content_id], [self.content.content_id])

    def update_param_changes(self, param_changes: List[param_domain.ParamChange]) -> None:
        if False:
            i = 10
            return i + 15
        'Update the param_changes dict attribute.\n\n        Args:\n            param_changes: list(ParamChange). List of param_change domain\n                objects that represents ParamChange domain object.\n        '
        self.param_changes = param_changes

    def update_interaction_id(self, interaction_id: Optional[str]) -> None:
        if False:
            return 10
        'Update the interaction id attribute.\n\n        Args:\n            interaction_id: str|None. The new interaction id to set.\n        '
        if self.interaction.id:
            old_content_id_list = [answer_group.outcome.feedback.content_id for answer_group in self.interaction.answer_groups]
            for answer_group in self.interaction.answer_groups:
                for rule_spec in answer_group.rule_specs:
                    for (param_name, value) in rule_spec.inputs.items():
                        param_type = interaction_registry.Registry.get_interaction_by_id(self.interaction.id).get_rule_param_type(rule_spec.rule_type, param_name)
                        if issubclass(param_type, objects.BaseTranslatableObject):
                            old_content_id_list.append(value['contentId'])
            self._update_content_ids_in_assets(old_content_id_list, [])
        self.interaction.id = interaction_id
        self.interaction.answer_groups = []

    def update_linked_skill_id(self, linked_skill_id: Optional[str]) -> None:
        if False:
            i = 10
            return i + 15
        'Update the state linked skill id attribute.\n\n        Args:\n            linked_skill_id: str|None. The linked skill id to state.\n        '
        self.linked_skill_id = linked_skill_id

    def update_interaction_customization_args(self, customization_args_mapping: Mapping[str, Mapping[str, UnionOfCustomizationArgsDictValues]]) -> None:
        if False:
            print('Hello World!')
        'Update the customization_args of InteractionInstance domain object.\n\n        Args:\n            customization_args_mapping: dict. The new customization_args to set.\n\n        Raises:\n            Exception. The customization arguments are not unique.\n        '
        customization_args_dict = cast(CustomizationArgsDictType, customization_args_mapping)
        customization_args = InteractionInstance.convert_customization_args_dict_to_customization_args(self.interaction.id, customization_args_dict)
        for ca_name in customization_args:
            customization_args[ca_name].validate_subtitled_html()
        old_content_id_list = list(itertools.chain.from_iterable([self.interaction.customization_args[ca_name].get_content_ids() for ca_name in self.interaction.customization_args]))
        self.interaction.customization_args = customization_args
        new_content_id_list = list(itertools.chain.from_iterable([self.interaction.customization_args[ca_name].get_content_ids() for ca_name in self.interaction.customization_args]))
        if len(new_content_id_list) != len(set(new_content_id_list)):
            raise Exception('All customization argument content_ids should be unique. Content ids received: %s' % new_content_id_list)
        self._update_content_ids_in_assets(old_content_id_list, new_content_id_list)

    def update_interaction_answer_groups(self, answer_groups_list: List[AnswerGroup]) -> None:
        if False:
            print('Hello World!')
        'Update the list of AnswerGroup in InteractionInstance domain object.\n\n        Args:\n            answer_groups_list: list(AnswerGroup). List of AnswerGroup domain\n                objects.\n\n        Raises:\n            Exception. Type of AnswerGroup domain objects is not as expected.\n        '
        if not isinstance(answer_groups_list, list):
            raise Exception('Expected interaction_answer_groups to be a list, received %s' % answer_groups_list)
        interaction_answer_groups = []
        new_content_id_list = []
        old_content_id_list = [answer_group.outcome.feedback.content_id for answer_group in self.interaction.answer_groups]
        for answer_group in self.interaction.answer_groups:
            for rule_spec in answer_group.rule_specs:
                for (param_name, value) in rule_spec.inputs.items():
                    param_type = interaction_registry.Registry.get_interaction_by_id(self.interaction.id).get_rule_param_type(rule_spec.rule_type, param_name)
                    if issubclass(param_type, objects.BaseTranslatableObject):
                        old_content_id_list.append(value['contentId'])
        for answer_group in answer_groups_list:
            rule_specs_list = answer_group.rule_specs
            if not isinstance(rule_specs_list, list):
                raise Exception('Expected answer group rule specs to be a list, received %s' % rule_specs_list)
            answer_group.rule_specs = []
            interaction_answer_groups.append(answer_group)
            for rule_spec in rule_specs_list:
                rule_inputs = rule_spec.inputs
                if not isinstance(rule_inputs, dict):
                    raise Exception('Expected rule_inputs to be a dict, received %s' % rule_inputs)
                for (param_name, value) in rule_inputs.items():
                    param_type = interaction_registry.Registry.get_interaction_by_id(self.interaction.id).get_rule_param_type(rule_spec.rule_type, param_name)
                    if isinstance(value, str) and '{{' in value and ('}}' in value):
                        normalized_param = value
                    else:
                        if issubclass(param_type, objects.BaseTranslatableObject):
                            new_content_id_list.append(value['contentId'])
                        try:
                            normalized_param = param_type.normalize(value)
                        except Exception as e:
                            raise Exception('Value has the wrong type. It should be a %s. The value is %s' % (param_type.__name__, value)) from e
                    rule_inputs[param_name] = normalized_param
                answer_group.rule_specs.append(rule_spec)
        self.interaction.answer_groups = interaction_answer_groups
        new_content_id_list += [answer_group.outcome.feedback.content_id for answer_group in self.interaction.answer_groups]
        self._update_content_ids_in_assets(old_content_id_list, new_content_id_list)

    def update_interaction_default_outcome(self, default_outcome: Optional[Outcome]) -> None:
        if False:
            return 10
        'Update the default_outcome of InteractionInstance domain object.\n\n        Args:\n            default_outcome: Outcome. Object representing the new Outcome.\n        '
        old_content_id_list = []
        new_content_id_list = []
        if self.interaction.default_outcome:
            old_content_id_list.append(self.interaction.default_outcome.feedback.content_id)
        if default_outcome:
            self.interaction.default_outcome = default_outcome
            new_content_id_list.append(self.interaction.default_outcome.feedback.content_id)
        else:
            self.interaction.default_outcome = None
        self._update_content_ids_in_assets(old_content_id_list, new_content_id_list)

    def update_interaction_confirmed_unclassified_answers(self, confirmed_unclassified_answers: List[AnswerGroup]) -> None:
        if False:
            i = 10
            return i + 15
        'Update the confirmed_unclassified_answers of IteractionInstance\n        domain object.\n\n        Args:\n            confirmed_unclassified_answers: list(AnswerGroup). The new list of\n                answers which have been confirmed to be associated with the\n                default outcome.\n\n        Raises:\n            Exception. Given answers is not of type list.\n        '
        if not isinstance(confirmed_unclassified_answers, list):
            raise Exception('Expected confirmed_unclassified_answers to be a list, received %s' % confirmed_unclassified_answers)
        self.interaction.confirmed_unclassified_answers = confirmed_unclassified_answers

    def update_interaction_hints(self, hints_list: List[Hint]) -> None:
        if False:
            while True:
                i = 10
        "Update the list of hints.\n\n        Args:\n            hints_list: list(Hint). A list of Hint objects.\n\n        Raises:\n            Exception. The 'hints_list' is not a list.\n        "
        if not isinstance(hints_list, list):
            raise Exception('Expected hints_list to be a list, received %s' % hints_list)
        old_content_id_list = [hint.hint_content.content_id for hint in self.interaction.hints]
        self.interaction.hints = copy.deepcopy(hints_list)
        new_content_id_list = [hint.hint_content.content_id for hint in self.interaction.hints]
        self._update_content_ids_in_assets(old_content_id_list, new_content_id_list)

    def update_interaction_solution(self, solution: Optional[Solution]) -> None:
        if False:
            while True:
                i = 10
        "Update the solution of interaction.\n\n        Args:\n            solution: Solution|None. Object of class Solution.\n\n        Raises:\n            Exception. The 'solution' is not a domain object.\n        "
        old_content_id_list = []
        new_content_id_list = []
        if self.interaction.solution:
            old_content_id_list.append(self.interaction.solution.explanation.content_id)
        if solution is not None:
            if not isinstance(solution, Solution):
                raise Exception('Expected solution to be a Solution object,received %s' % solution)
            self.interaction.solution = solution
            new_content_id_list.append(self.interaction.solution.explanation.content_id)
        else:
            self.interaction.solution = None
        self._update_content_ids_in_assets(old_content_id_list, new_content_id_list)

    def update_recorded_voiceovers(self, recorded_voiceovers: RecordedVoiceovers) -> None:
        if False:
            i = 10
            return i + 15
        'Update the recorded_voiceovers of a state.\n\n        Args:\n            recorded_voiceovers: RecordedVoiceovers. The new RecordedVoiceovers\n                object for the state.\n        '
        self.recorded_voiceovers = recorded_voiceovers

    def update_solicit_answer_details(self, solicit_answer_details: bool) -> None:
        if False:
            while True:
                i = 10
        'Update the solicit_answer_details of a state.\n\n        Args:\n            solicit_answer_details: bool. The new value of\n                solicit_answer_details for the state.\n\n        Raises:\n            Exception. The argument is not of type bool.\n        '
        if not isinstance(solicit_answer_details, bool):
            raise Exception('Expected solicit_answer_details to be a boolean, received %s' % solicit_answer_details)
        self.solicit_answer_details = solicit_answer_details

    def update_card_is_checkpoint(self, card_is_checkpoint: bool) -> None:
        if False:
            while True:
                i = 10
        'Update the card_is_checkpoint field of a state.\n\n        Args:\n            card_is_checkpoint: bool. The new value of\n                card_is_checkpoint for the state.\n\n        Raises:\n            Exception. The argument is not of type bool.\n        '
        if not isinstance(card_is_checkpoint, bool):
            raise Exception('Expected card_is_checkpoint to be a boolean, received %s' % card_is_checkpoint)
        self.card_is_checkpoint = card_is_checkpoint

    def to_dict(self) -> StateDict:
        if False:
            print('Hello World!')
        'Returns a dict representing this State domain object.\n\n        Returns:\n            dict. A dict mapping all fields of State instance.\n        '
        return {'content': self.content.to_dict(), 'param_changes': [param_change.to_dict() for param_change in self.param_changes], 'interaction': self.interaction.to_dict(), 'classifier_model_id': self.classifier_model_id, 'linked_skill_id': self.linked_skill_id, 'recorded_voiceovers': self.recorded_voiceovers.to_dict(), 'solicit_answer_details': self.solicit_answer_details, 'card_is_checkpoint': self.card_is_checkpoint}

    @classmethod
    def from_dict(cls, state_dict: StateDict, validate: bool=True) -> State:
        if False:
            i = 10
            return i + 15
        'Return a State domain object from a dict.\n\n        Args:\n            state_dict: dict. The dict representation of State object.\n            validate: bool. False, when the validations should not be called.\n\n        Returns:\n            State. The corresponding State domain object.\n        '
        content = SubtitledHtml.from_dict(state_dict['content'])
        if validate:
            content.validate()
        return cls(content, [param_domain.ParamChange.from_dict(param) for param in state_dict['param_changes']], InteractionInstance.from_dict(state_dict['interaction'], validate=validate), RecordedVoiceovers.from_dict(state_dict['recorded_voiceovers']), state_dict['solicit_answer_details'], state_dict['card_is_checkpoint'], state_dict['linked_skill_id'], state_dict['classifier_model_id'])

    @classmethod
    def create_default_state(cls, default_dest_state_name: Optional[str], content_id_for_state_content: str, content_id_for_default_outcome: str, is_initial_state: bool=False) -> State:
        if False:
            return 10
        'Return a State domain object with default value.\n\n        Args:\n            default_dest_state_name: str|None. The default destination state, or\n                None if no default destination state is defined.\n            is_initial_state: bool. Whether this state represents the initial\n                state of an exploration.\n            content_id_for_state_content: str. The content id for the content.\n            content_id_for_default_outcome: str. The content id for the default\n                outcome.\n\n        Returns:\n            State. The corresponding State domain object.\n        '
        content_html = feconf.DEFAULT_INIT_STATE_CONTENT_STR if is_initial_state else ''
        recorded_voiceovers = RecordedVoiceovers({})
        recorded_voiceovers.add_content_id_for_voiceover(content_id_for_state_content)
        recorded_voiceovers.add_content_id_for_voiceover(content_id_for_default_outcome)
        return cls(SubtitledHtml(content_id_for_state_content, content_html), [], InteractionInstance.create_default_interaction(default_dest_state_name, content_id_for_default_outcome), recorded_voiceovers, False, is_initial_state)

    @classmethod
    def convert_html_fields_in_state(cls, state_dict: StateDict, conversion_fn: Callable[[str], str], state_schema_version: int=feconf.CURRENT_STATE_SCHEMA_VERSION, state_uses_old_interaction_cust_args_schema: bool=False, state_uses_old_rule_template_schema: bool=False) -> StateDict:
        if False:
            return 10
        'Applies a conversion function on all the html strings in a state\n        to migrate them to a desired state.\n\n        Args:\n            state_dict: dict. The dict representation of State object.\n            conversion_fn: function. The conversion function to be applied on\n                the states_dict.\n            state_schema_version: int. The state schema version.\n            state_uses_old_interaction_cust_args_schema: bool. Whether the\n                interaction customization arguments contain SubtitledHtml\n                and SubtitledUnicode dicts (should be True if prior to state\n                schema v36).\n            state_uses_old_rule_template_schema: bool. Whether the rule inputs\n                contain html in the form of DragAndDropHtmlString,\n                SetOfHtmlString, or ListOfSetsOfHtmlString (shoud be True if\n                prior to state schema v42).\n\n        Returns:\n            dict. The converted state_dict.\n        '
        state_dict['content']['html'] = conversion_fn(state_dict['content']['html'])
        if state_dict['interaction']['default_outcome'] is not None:
            state_dict['interaction']['default_outcome'] = Outcome.convert_html_in_outcome(state_dict['interaction']['default_outcome'], conversion_fn)
        if state_uses_old_rule_template_schema:
            html_field_types_to_rule_specs = rules_registry.Registry.get_html_field_types_to_rule_specs(state_schema_version=41)
        else:
            html_field_types_to_rule_specs = rules_registry.Registry.get_html_field_types_to_rule_specs()
        for (answer_group_index, answer_group) in enumerate(state_dict['interaction']['answer_groups']):
            state_dict['interaction']['answer_groups'][answer_group_index] = AnswerGroup.convert_html_in_answer_group(answer_group, conversion_fn, html_field_types_to_rule_specs)
        for (hint_index, hint) in enumerate(state_dict['interaction']['hints']):
            state_dict['interaction']['hints'][hint_index] = Hint.convert_html_in_hint(hint, conversion_fn)
        interaction_id = state_dict['interaction']['id']
        all_interaction_ids = interaction_registry.Registry.get_all_interaction_ids()
        interaction_id_is_valid = interaction_id not in all_interaction_ids
        if interaction_id_is_valid or interaction_id is None:
            return state_dict
        if state_dict['interaction']['solution'] is not None:
            if state_uses_old_rule_template_schema:
                interaction_spec = interaction_registry.Registry.get_all_specs_for_state_schema_version(41)[interaction_id]
            else:
                interaction_spec = interaction_registry.Registry.get_all_specs()[interaction_id]
            state_dict['interaction']['solution'] = Solution.convert_html_in_solution(state_dict['interaction']['id'], state_dict['interaction']['solution'], conversion_fn, html_field_types_to_rule_specs, interaction_spec)
        if state_uses_old_interaction_cust_args_schema:
            ca_specs = interaction_registry.Registry.get_all_specs_for_state_schema_version(35)[interaction_id]['customization_arg_specs']
            interaction_customization_arg_has_html = False
            for customization_arg_spec in ca_specs:
                schema = customization_arg_spec['schema']
                if schema['type'] == schema_utils.SCHEMA_TYPE_LIST and schema['items']['type'] == schema_utils.SCHEMA_TYPE_HTML:
                    interaction_customization_arg_has_html = True
            if interaction_customization_arg_has_html:
                if 'choices' in state_dict['interaction']['customization_args'].keys():
                    html_choices_ca_dict = cast(Dict[str, List[str]], state_dict['interaction']['customization_args']['choices'])
                    html_choices_ca_dict['value'] = [conversion_fn(html) for html in html_choices_ca_dict['value']]
        else:
            ca_specs_dict = interaction_registry.Registry.get_all_specs_for_state_schema_version(state_schema_version, can_fetch_latest_specs=True)[interaction_id]['customization_arg_specs']
            state_dict['interaction'] = InteractionInstance.convert_html_in_interaction(state_dict['interaction'], ca_specs_dict, conversion_fn)
        return state_dict

    def get_content_html(self, content_id: str) -> Union[str, List[str]]:
        if False:
            while True:
                i = 10
        'Returns the content belongs to a given content id of the object.\n\n        Args:\n            content_id: str. The id of the content.\n\n        Returns:\n            str. The html content corresponding to the given content id.\n\n        Raises:\n            ValueError. The given content_id does not exist.\n        '
        content_id_to_translatable_content = self.get_translatable_contents_collection().content_id_to_translatable_content
        if content_id not in content_id_to_translatable_content:
            raise ValueError('Content ID %s does not exist' % content_id)
        return content_id_to_translatable_content[content_id].content_value

    @classmethod
    def traverse_v54_state_dict_for_contents(cls, state_dict: StateDict) -> Iterator[Tuple[Union[SubtitledHtmlDict, Dict[str, Union[str, List[str]]]], translation_domain.ContentType, Optional[str]]]:
        if False:
            for i in range(10):
                print('nop')
        'This method iterates throughout the state dict and yields the value\n        for each field. The yielded value is used for generating and updating\n        the content-ids for the fields in the state in their respective methods.\n\n        Args:\n            state_dict: StateDict. State object represented in the dict format.\n\n        Yields:\n            (str|list(str), str). A tuple containing content and content-id.\n        '
        yield (state_dict['content'], translation_domain.ContentType.CONTENT, None)
        interaction = state_dict['interaction']
        default_outcome = interaction['default_outcome']
        if default_outcome is not None:
            yield (default_outcome['feedback'], translation_domain.ContentType.DEFAULT_OUTCOME, None)
        answer_groups = interaction['answer_groups']
        for answer_group in answer_groups:
            outcome = answer_group['outcome']
            yield (outcome['feedback'], translation_domain.ContentType.FEEDBACK, None)
            if interaction['id'] not in ['TextInput', 'SetInput']:
                continue
            for rule_spec in answer_group['rule_specs']:
                for input_name in sorted(rule_spec['inputs'].keys()):
                    input_value = rule_spec['inputs'][input_name]
                    if not isinstance(input_value, dict):
                        continue
                    if 'normalizedStrSet' in input_value:
                        yield (input_value, translation_domain.ContentType.RULE, 'input')
                    if 'unicodeStrSet' in input_value:
                        yield (input_value, translation_domain.ContentType.RULE, 'input')
        for hint in interaction['hints']:
            yield (hint['hint_content'], translation_domain.ContentType.HINT, None)
        solution = interaction['solution']
        if solution is not None:
            yield (solution['explanation'], translation_domain.ContentType.SOLUTION, None)
        interaction_id = interaction['id']
        customisation_args = interaction['customization_args']
        interaction_specs = interaction_registry.Registry.get_all_specs_for_state_schema_version(feconf.CURRENT_STATE_SCHEMA_VERSION, can_fetch_latest_specs=True)
        if interaction_id in interaction_specs:
            ca_specs_dict = interaction_specs[interaction_id]['customization_arg_specs']
            for spec in ca_specs_dict:
                if spec['name'] != 'catchMisspellings':
                    customisation_arg = customisation_args[spec['name']]
                    contents = InteractionCustomizationArg.traverse_by_schema_and_get(spec['schema'], customisation_arg['value'], [schema_utils.SCHEMA_OBJ_TYPE_SUBTITLED_UNICODE, schema_utils.SCHEMA_OBJ_TYPE_SUBTITLED_HTML], lambda x: x)
                    for content in contents:
                        yield (content, translation_domain.ContentType.CUSTOMIZATION_ARG, spec['name'])

    @classmethod
    def update_old_content_id_to_new_content_id_in_v54_states(cls, states_dict: Dict[str, StateDict]) -> Tuple[Dict[str, StateDict], int]:
        if False:
            for i in range(10):
                print('nop')
        'Updates the old content-ids from the state fields like hints,\n        solution, etc with the newly generated content id.\n\n        Args:\n            states_dict: list(dict(State)). List of dictionaries, where each\n                dict represents a state object.\n\n        Returns:\n            states_dict: list(dict(State)). List of state dicts, with updated\n            content-ids.\n        '
        PossibleContentIdsType = Union[str, List[str], List[List[str]]]

        def _replace_content_id(old_id: PossibleContentIdsType, id_mapping: Dict[str, str]) -> str:
            if False:
                print('Hello World!')
            'Replace old Id with the new Id.'
            assert isinstance(old_id, str)
            if old_id == feconf.INVALID_CONTENT_ID:
                return old_id
            return id_mapping[old_id]
        object_content_ids_replacers: Dict[str, Callable[[PossibleContentIdsType, Dict[str, str]], PossibleContentIdsType]] = {}
        object_content_ids_replacers['TranslatableHtmlContentId'] = _replace_content_id
        object_content_ids_replacers['SetOfTranslatableHtmlContentIds'] = lambda ids_set, id_mapping: [_replace_content_id(old_id, id_mapping) for old_id in ids_set]
        object_content_ids_replacers['ListOfSetsOfTranslatableHtmlContentIds'] = lambda items, id_mapping: [[_replace_content_id(old_id, id_mapping) for old_id in ids_set] for ids_set in items]
        content_id_generator = translation_domain.ContentIdGenerator()
        for state_name in sorted(states_dict.keys()):
            state: StateDict = states_dict[state_name]
            new_voiceovers_mapping: Dict[str, Dict[str, VoiceoverDict]] = {}
            old_to_new_content_id: Dict[str, str] = {}
            old_voiceovers_mapping = state['recorded_voiceovers']['voiceovers_mapping']
            for (content, content_type, extra_prefix) in cls.traverse_v54_state_dict_for_contents(state):
                new_content_id = content_id_generator.generate(content_type, extra_prefix=extra_prefix)
                content_id_key = 'content_id'
                if content_type == translation_domain.ContentType.RULE:
                    content_id_key = 'contentId'
                old_content_id = content[content_id_key]
                content[content_id_key] = new_content_id
                assert isinstance(old_content_id, str)
                old_to_new_content_id[old_content_id] = new_content_id
                new_voiceovers_mapping[new_content_id] = old_voiceovers_mapping[old_content_id]
            state['recorded_voiceovers']['voiceovers_mapping'] = new_voiceovers_mapping
            interaction_specs = interaction_registry.Registry.get_all_specs_for_state_schema_version(feconf.CURRENT_STATE_SCHEMA_VERSION, can_fetch_latest_specs=True)
            interaction_id = state['interaction']['id']
            if interaction_id is None:
                continue
            interaction = state['interaction']
            answer_groups = interaction['answer_groups']
            rule_descriptions = interaction_specs[interaction_id]['rule_descriptions']
            answer_type = interaction_specs[interaction_id]['answer_type']
            if interaction['solution']:
                solution_dict = interaction['solution']
                assert solution_dict is not None
                if answer_type in object_content_ids_replacers:
                    correct_answer = cast(PossibleContentIdsType, solution_dict['correct_answer'])
                    solution_dict['correct_answer'] = object_content_ids_replacers[answer_type](correct_answer, old_to_new_content_id)
            if not rule_descriptions:
                continue
            rules_variables = {name: re.findall('\\{\\{(.+?)\\|(.+?)\\}\\}', description) for (name, description) in rule_descriptions.items()}
            for answer_group in answer_groups:
                for rule_spec in answer_group['rule_specs']:
                    rule_inputs = rule_spec['inputs']
                    rule_type = rule_spec['rule_type']
                    for (key, value_class) in rules_variables[rule_type]:
                        if value_class not in object_content_ids_replacers:
                            continue
                        rule_input = cast(PossibleContentIdsType, rule_inputs[key])
                        rule_inputs[key] = object_content_ids_replacers[value_class](rule_input, old_to_new_content_id)
        return (states_dict, content_id_generator.next_content_id_index)

    @classmethod
    def generate_old_content_id_to_new_content_id_in_v54_states(cls, states_dict: Dict[str, StateDict]) -> Tuple[Dict[str, Dict[str, str]], int]:
        if False:
            print('Hello World!')
        'Generates the new content-id for each state field based on\n        next_content_id_index variable.\n\n        Args:\n            states_dict: list(dict(State)). List of dictionaries, where each\n                dict represents a state object.\n\n        Returns:\n            (dict(str, dict(str, str)), str). A tuple with the first field as a\n            dict and the second field is the value of the next_content_id_index.\n            The first field is a dict with state name as a key and\n            old-content-id to new-content-id dict as a value.\n        '
        content_id_generator = translation_domain.ContentIdGenerator()
        states_to_content_id = {}
        for state_name in sorted(states_dict.keys()):
            old_id_to_new_id: Dict[str, str] = {}
            for (content, content_type, extra_prefix) in cls.traverse_v54_state_dict_for_contents(states_dict[state_name]):
                if content_type == translation_domain.ContentType.RULE:
                    content_id = content['contentId']
                else:
                    content_id = content['content_id']
                assert isinstance(content_id, str)
                old_id_to_new_id[content_id] = content_id_generator.generate(content_type, extra_prefix=extra_prefix)
            states_to_content_id[state_name] = old_id_to_new_id
        return (states_to_content_id, content_id_generator.next_content_id_index)

class StateVersionHistory:
    """Class to represent an element of the version history list of a state.
    The version history list of a state is the list of exploration versions
    in which the state has been edited.

    Attributes:
        previously_edited_in_version: int. The version number of the
            exploration in which the state was previously edited.
        state_name_in_previous_version: str. The name of the state in the
            previously edited version. It is useful in case of state renames.
        committer_id: str. The id of the user who committed the changes in the
            previously edited version.
    """

    def __init__(self, previously_edited_in_version: Optional[int], state_name_in_previous_version: Optional[str], committer_id: str) -> None:
        if False:
            print('Hello World!')
        'Initializes the StateVersionHistory domain object.\n\n        Args:\n            previously_edited_in_version: int. The version number of the\n                exploration on which the state was previously edited.\n            state_name_in_previous_version: str. The name of the state in the\n                previously edited version. It is useful in case of state\n                renames.\n            committer_id: str. The id of the user who committed the changes in\n                the previously edited version.\n        '
        self.previously_edited_in_version = previously_edited_in_version
        self.state_name_in_previous_version = state_name_in_previous_version
        self.committer_id = committer_id

    def to_dict(self) -> StateVersionHistoryDict:
        if False:
            i = 10
            return i + 15
        'Returns a dict representation of the StateVersionHistory domain\n        object.\n\n        Returns:\n            dict. The dict representation of the StateVersionHistory domain\n            object.\n        '
        return {'previously_edited_in_version': self.previously_edited_in_version, 'state_name_in_previous_version': self.state_name_in_previous_version, 'committer_id': self.committer_id}

    @classmethod
    def from_dict(cls, state_version_history_dict: StateVersionHistoryDict) -> StateVersionHistory:
        if False:
            return 10
        'Return a StateVersionHistory domain object from a dict.\n\n        Args:\n            state_version_history_dict: dict. The dict representation of\n                StateVersionHistory object.\n\n        Returns:\n            StateVersionHistory. The corresponding StateVersionHistory domain\n            object.\n        '
        return cls(state_version_history_dict['previously_edited_in_version'], state_version_history_dict['state_name_in_previous_version'], state_version_history_dict['committer_id'])
UnionOfCustomizationArgsDictValues = Union[str, int, bool, List[str], List[SubtitledHtml], List[SubtitledHtmlDict], SubtitledHtmlDict, SubtitledUnicode, SubtitledUnicodeDict, domain.ImageAndRegionDict, domain.GraphDict]
CustomizationArgsDictType = Dict[str, Dict[str, UnionOfCustomizationArgsDictValues]]