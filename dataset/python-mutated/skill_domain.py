"""Domain objects relating to skills."""
from __future__ import annotations
import copy
import datetime
import json
from core import android_validation_constants
from core import feconf
from core import utils
from core.constants import constants
from core.domain import change_domain
from core.domain import state_domain
from core.domain import translation_domain
from typing import Callable, Dict, Final, List, Literal, Optional, TypedDict
from core.domain import html_cleaner
from core.domain import html_validation_service
SKILL_PROPERTY_DESCRIPTION: Final = 'description'
SKILL_PROPERTY_LANGUAGE_CODE: Final = 'language_code'
SKILL_PROPERTY_SUPERSEDING_SKILL_ID: Final = 'superseding_skill_id'
SKILL_PROPERTY_ALL_QUESTIONS_MERGED: Final = 'all_questions_merged'
SKILL_PROPERTY_PREREQUISITE_SKILL_IDS: Final = 'prerequisite_skill_ids'
SKILL_CONTENTS_PROPERTY_EXPLANATION: Final = 'explanation'
SKILL_CONTENTS_PROPERTY_WORKED_EXAMPLES: Final = 'worked_examples'
SKILL_MISCONCEPTIONS_PROPERTY_NAME: Final = 'name'
SKILL_MISCONCEPTIONS_PROPERTY_NOTES: Final = 'notes'
SKILL_MISCONCEPTIONS_PROPERTY_FEEDBACK: Final = 'feedback'
SKILL_MISCONCEPTIONS_PROPERTY_MUST_BE_ADDRESSED: Final = 'must_be_addressed'
CMD_UPDATE_SKILL_PROPERTY: Final = 'update_skill_property'
CMD_UPDATE_SKILL_CONTENTS_PROPERTY: Final = 'update_skill_contents_property'
CMD_UPDATE_SKILL_MISCONCEPTIONS_PROPERTY: Final = 'update_skill_misconceptions_property'
CMD_UPDATE_RUBRICS: Final = 'update_rubrics'
CMD_ADD_SKILL_MISCONCEPTION: Final = 'add_skill_misconception'
CMD_DELETE_SKILL_MISCONCEPTION: Final = 'delete_skill_misconception'
CMD_ADD_PREREQUISITE_SKILL: Final = 'add_prerequisite_skill'
CMD_DELETE_PREREQUISITE_SKILL: Final = 'delete_prerequisite_skill'
CMD_CREATE_NEW: Final = 'create_new'
CMD_MIGRATE_CONTENTS_SCHEMA_TO_LATEST_VERSION: Final = 'migrate_contents_schema_to_latest_version'
CMD_MIGRATE_MISCONCEPTIONS_SCHEMA_TO_LATEST_VERSION: Final = 'migrate_misconceptions_schema_to_latest_version'
CMD_MIGRATE_RUBRICS_SCHEMA_TO_LATEST_VERSION: Final = 'migrate_rubrics_schema_to_latest_version'

class SkillChange(change_domain.BaseChange):
    """Domain object for changes made to skill object.

    The allowed commands, together with the attributes:
        - 'add_skill_misconception' (with new_misconception_dict)
        - 'delete_skill_misconception' (with misconception_id)
        - 'create_new'
        - 'update_skill_property' (with property_name, new_value
        and old_value)
        - 'update_skill_contents_property' (with property_name,
        new_value and old_value)
        - 'update_skill_misconceptions_property' (
            with misconception_id, property_name, new_value and old_value)
        - 'migrate_contents_schema_to_latest_version' (with
        from_version and to_version)
        - 'migrate_misconceptions_schema_to_latest_version' (with
        from_version and to_version)
    """
    SKILL_PROPERTIES: List[str] = [SKILL_PROPERTY_DESCRIPTION, SKILL_PROPERTY_LANGUAGE_CODE, SKILL_PROPERTY_SUPERSEDING_SKILL_ID, SKILL_PROPERTY_ALL_QUESTIONS_MERGED, SKILL_PROPERTY_PREREQUISITE_SKILL_IDS]
    SKILL_CONTENTS_PROPERTIES: List[str] = [SKILL_CONTENTS_PROPERTY_EXPLANATION, SKILL_CONTENTS_PROPERTY_WORKED_EXAMPLES]
    SKILL_MISCONCEPTIONS_PROPERTIES: List[str] = [SKILL_MISCONCEPTIONS_PROPERTY_NAME, SKILL_MISCONCEPTIONS_PROPERTY_NOTES, SKILL_MISCONCEPTIONS_PROPERTY_FEEDBACK, SKILL_MISCONCEPTIONS_PROPERTY_MUST_BE_ADDRESSED]
    ALLOWED_COMMANDS: List[feconf.ValidCmdDict] = [{'name': CMD_CREATE_NEW, 'required_attribute_names': [], 'optional_attribute_names': [], 'user_id_attribute_names': [], 'allowed_values': {}, 'deprecated_values': {}}, {'name': CMD_ADD_SKILL_MISCONCEPTION, 'required_attribute_names': ['new_misconception_dict'], 'optional_attribute_names': [], 'user_id_attribute_names': [], 'allowed_values': {}, 'deprecated_values': {}}, {'name': CMD_DELETE_SKILL_MISCONCEPTION, 'required_attribute_names': ['misconception_id'], 'optional_attribute_names': [], 'user_id_attribute_names': [], 'allowed_values': {}, 'deprecated_values': {}}, {'name': CMD_ADD_PREREQUISITE_SKILL, 'required_attribute_names': ['skill_id'], 'optional_attribute_names': [], 'user_id_attribute_names': [], 'allowed_values': {}, 'deprecated_values': {}}, {'name': CMD_DELETE_PREREQUISITE_SKILL, 'required_attribute_names': ['skill_id'], 'optional_attribute_names': [], 'user_id_attribute_names': [], 'allowed_values': {}, 'deprecated_values': {}}, {'name': CMD_UPDATE_RUBRICS, 'required_attribute_names': ['difficulty', 'explanations'], 'optional_attribute_names': [], 'user_id_attribute_names': [], 'allowed_values': {}, 'deprecated_values': {}}, {'name': CMD_UPDATE_SKILL_MISCONCEPTIONS_PROPERTY, 'required_attribute_names': ['misconception_id', 'property_name', 'new_value', 'old_value'], 'optional_attribute_names': [], 'user_id_attribute_names': [], 'allowed_values': {'property_name': SKILL_MISCONCEPTIONS_PROPERTIES}, 'deprecated_values': {}}, {'name': CMD_UPDATE_SKILL_PROPERTY, 'required_attribute_names': ['property_name', 'new_value', 'old_value'], 'optional_attribute_names': [], 'user_id_attribute_names': [], 'allowed_values': {'property_name': SKILL_PROPERTIES}, 'deprecated_values': {}}, {'name': CMD_UPDATE_SKILL_CONTENTS_PROPERTY, 'required_attribute_names': ['property_name', 'new_value', 'old_value'], 'optional_attribute_names': [], 'user_id_attribute_names': [], 'allowed_values': {'property_name': SKILL_CONTENTS_PROPERTIES}, 'deprecated_values': {}}, {'name': CMD_MIGRATE_CONTENTS_SCHEMA_TO_LATEST_VERSION, 'required_attribute_names': ['from_version', 'to_version'], 'optional_attribute_names': [], 'user_id_attribute_names': [], 'allowed_values': {}, 'deprecated_values': {}}, {'name': CMD_MIGRATE_MISCONCEPTIONS_SCHEMA_TO_LATEST_VERSION, 'required_attribute_names': ['from_version', 'to_version'], 'optional_attribute_names': [], 'user_id_attribute_names': [], 'allowed_values': {}, 'deprecated_values': {}}, {'name': CMD_MIGRATE_RUBRICS_SCHEMA_TO_LATEST_VERSION, 'required_attribute_names': ['from_version', 'to_version'], 'optional_attribute_names': [], 'user_id_attribute_names': [], 'allowed_values': {}, 'deprecated_values': {}}]

class CreateNewSkillCmd(SkillChange):
    """Class representing the SkillChange's
    CMD_CREATE_NEW command.
    """
    pass

class AddSkillMisconceptionCmd(SkillChange):
    """Class representing the SkillChange's
    CMD_ADD_SKILL_MISCONCEPTION command.
    """
    new_misconception_dict: MisconceptionDict

class DeleteSkillMisconceptionCmd(SkillChange):
    """Class representing the SkillChange's
    CMD_DELETE_SKILL_MISCONCEPTION command.
    """
    misconception_id: int

class AddPrerequisiteSkillCmd(SkillChange):
    """Class representing the SkillChange's
    CMD_ADD_PREREQUISITE_SKILL command.
    """
    skill_id: str

class DeletePrerequisiteSkillCmd(SkillChange):
    """Class representing the SkillChange's
    CMD_DELETE_PREREQUISITE_SKILL command.
    """
    skill_id: str

class UpdateRubricsCmd(SkillChange):
    """Class representing the SkillChange's
    CMD_UPDATE_RUBRICS command.
    """
    difficulty: str
    explanations: List[str]

class UpdateSkillMisconceptionPropertyNameCmd(SkillChange):
    """Class representing the SkillChange's
    CMD_UPDATE_SKILL_MISCONCEPTIONS_PROPERTY command with
    SKILL_MISCONCEPTIONS_PROPERTY_NAME as allowed value.
    """
    misconception_id: int
    property_name: Literal['name']
    new_value: str
    old_value: str

class UpdateSkillMisconceptionPropertyNotesCmd(SkillChange):
    """Class representing the SkillChange's
    CMD_UPDATE_SKILL_MISCONCEPTIONS_PROPERTY command with
    SKILL_MISCONCEPTIONS_PROPERTY_NOTES as allowed value.
    """
    misconception_id: int
    property_name: Literal['notes']
    new_value: str
    old_value: str

class UpdateSkillMisconceptionPropertyFeedbackCmd(SkillChange):
    """Class representing the SkillChange's
    CMD_UPDATE_SKILL_MISCONCEPTIONS_PROPERTY command with
    SKILL_MISCONCEPTIONS_PROPERTY_FEEDBACK as allowed value.
    """
    misconception_id: int
    property_name: Literal['feedback']
    new_value: str
    old_value: str

class UpdateSkillMisconceptionPropertyMustBeAddressedCmd(SkillChange):
    """Class representing the SkillChange's
    CMD_UPDATE_SKILL_MISCONCEPTIONS_PROPERTY command with
    SKILL_MISCONCEPTIONS_PROPERTY_MUST_BE_ADDRESSED as allowed value.
    """
    misconception_id: int
    property_name: Literal['must_be_addressed']
    new_value: bool
    old_value: bool

class UpdateSkillPropertyDescriptionCmd(SkillChange):
    """Class representing the SkillChange's
    CMD_UPDATE_SKILL_PROPERTY command with
    SKILL_PROPERTY_DESCRIPTION as allowed value.
    """
    property_name: Literal['description']
    new_value: str
    old_value: str

class UpdateSkillPropertyLanguageCodeCmd(SkillChange):
    """Class representing the SkillChange's
    CMD_UPDATE_SKILL_PROPERTY command with
    SKILL_PROPERTY_LANGUAGE_CODE as allowed value.
    """
    property_name: Literal['language_code']
    new_value: str
    old_value: str

class UpdateSkillPropertySupersedingSkillIdCmd(SkillChange):
    """Class representing the SkillChange's
    CMD_UPDATE_SKILL_PROPERTY command with
    SKILL_PROPERTY_SUPERSEDING_SKILL_ID as
    allowed value.
    """
    property_name: Literal['superseding_skill_id']
    new_value: str
    old_value: str

class UpdateSkillPropertyAllQuestionsMergedCmd(SkillChange):
    """Class representing the SkillChange's
    CMD_UPDATE_SKILL_PROPERTY command with
    SKILL_PROPERTY_ALL_QUESTIONS_MERGED as
    allowed value.
    """
    property_name: Literal['all_questions_merged']
    new_value: bool
    old_value: bool

class UpdateSkillPropertyPrerequisiteSkillIdsCmd(SkillChange):
    """Class representing the SkillChange's
    CMD_UPDATE_SKILL_PROPERTY command with
    SKILL_PROPERTY_PREREQUISITE_SKILL_IDS as
    allowed value.
    """
    property_name: Literal['prerequisite_skill_ids']
    new_value: List[str]
    old_value: List[str]

class UpdateSkillContentsPropertyExplanationCmd(SkillChange):
    """Class representing the SkillChange's
    CMD_UPDATE_SKILL_CONTENTS_PROPERTY command
    with SKILL_CONTENTS_PROPERTY_EXPLANATION as
    allowed value.
    """
    property_name: Literal['explanation']
    new_value: state_domain.SubtitledHtmlDict
    old_value: state_domain.SubtitledHtmlDict

class UpdateSkillContentsPropertyWorkedExamplesCmd(SkillChange):
    """Class representing the SkillChange's
    CMD_UPDATE_SKILL_CONTENTS_PROPERTY command
    with SKILL_CONTENTS_PROPERTY_WORKED_EXAMPLES
    as allowed value.
    """
    property_name: Literal['worked_examples']
    new_value: List[WorkedExampleDict]
    old_value: List[WorkedExampleDict]

class MigrateContentsSchemaToLatestVersionCmd(SkillChange):
    """Class representing the SkillChange's
    CMD_MIGRATE_CONTENTS_SCHEMA_TO_LATEST_VERSION command.
    """
    from_version: str
    to_version: str

class MigrateMisconceptionsSchemaToLatestVersionCmd(SkillChange):
    """Class representing the SkillChange's
    CMD_MIGRATE_MISCONCEPTIONS_SCHEMA_TO_LATEST_VERSION command.
    """
    from_version: str
    to_version: str

class MigrateRubricsSchemaToLatestVersionCmd(SkillChange):
    """Class representing the SkillChange's
    CMD_MIGRATE_MISCONCEPTIONS_SCHEMA_TO_LATEST_VERSION command.
    """
    from_version: str
    to_version: str

class MisconceptionDict(TypedDict):
    """Dictionary representing the Misconception object."""
    id: int
    name: str
    notes: str
    feedback: str
    must_be_addressed: bool

class VersionedMisconceptionDict(TypedDict):
    """Dictionary representing the versioned Misconception object."""
    schema_version: int
    misconceptions: List[MisconceptionDict]

class Misconception:
    """Domain object describing a skill misconception."""

    def __init__(self, misconception_id: int, name: str, notes: str, feedback: str, must_be_addressed: bool) -> None:
        if False:
            print('Hello World!')
        'Initializes a Misconception domain object.\n\n        Args:\n            misconception_id: int. The unique id of each misconception.\n            name: str. The name of the misconception.\n            notes: str. General advice for creators about the\n                misconception (including examples) and general notes. This\n                should be an html string.\n            feedback: str. This can auto-populate the feedback field\n                when an answer group has been tagged with a misconception. This\n                should be an html string.\n            must_be_addressed: bool. Whether the misconception should\n                necessarily be addressed in all questions linked to the skill.\n        '
        self.id = misconception_id
        self.name = name
        self.notes = html_cleaner.clean(notes)
        self.feedback = html_cleaner.clean(feedback)
        self.must_be_addressed = must_be_addressed

    def to_dict(self) -> MisconceptionDict:
        if False:
            while True:
                i = 10
        'Returns a dict representing this Misconception domain object.\n\n        Returns:\n            dict. A dict, mapping all fields of Misconception instance.\n        '
        return {'id': self.id, 'name': self.name, 'notes': self.notes, 'feedback': self.feedback, 'must_be_addressed': self.must_be_addressed}

    @classmethod
    def from_dict(cls, misconception_dict: MisconceptionDict) -> Misconception:
        if False:
            print('Hello World!')
        'Returns a Misconception domain object from a dict.\n\n        Args:\n            misconception_dict: dict. The dict representation of\n                Misconception object.\n\n        Returns:\n            Misconception. The corresponding Misconception domain object.\n        '
        misconception = cls(misconception_dict['id'], misconception_dict['name'], misconception_dict['notes'], misconception_dict['feedback'], misconception_dict['must_be_addressed'])
        return misconception

    @classmethod
    def require_valid_misconception_id(cls, misconception_id: int) -> None:
        if False:
            return 10
        'Validates the misconception id for a Misconception object.\n\n        Args:\n            misconception_id: int. The misconception id to be validated.\n\n        Raises:\n            ValidationError. The misconception id is invalid.\n        '
        if not isinstance(misconception_id, int):
            raise utils.ValidationError('Expected misconception ID to be an integer, received %s' % misconception_id)
        if misconception_id < 0:
            raise utils.ValidationError('Expected misconception ID to be >= 0, received %s' % misconception_id)

    def validate(self) -> None:
        if False:
            print('Hello World!')
        'Validates various properties of the Misconception object.\n\n        Raises:\n            ValidationError. One or more attributes of the misconception are\n                invalid.\n        '
        self.require_valid_misconception_id(self.id)
        if not isinstance(self.name, str):
            raise utils.ValidationError('Expected misconception name to be a string, received %s' % self.name)
        misconception_name_length_limit = android_validation_constants.MAX_CHARS_IN_MISCONCEPTION_NAME
        if len(self.name) > misconception_name_length_limit:
            raise utils.ValidationError('Misconception name should be less than %d chars, received %s' % (misconception_name_length_limit, self.name))
        if not isinstance(self.notes, str):
            raise utils.ValidationError('Expected misconception notes to be a string, received %s' % self.notes)
        if not isinstance(self.must_be_addressed, bool):
            raise utils.ValidationError('Expected must_be_addressed to be a bool, received %s' % self.must_be_addressed)
        if not isinstance(self.feedback, str):
            raise utils.ValidationError('Expected misconception feedback to be a string, received %s' % self.feedback)

class RubricDict(TypedDict):
    """Dictionary representing the Rubric object."""
    difficulty: str
    explanations: List[str]

class VersionedRubricDict(TypedDict):
    """Dictionary representing the versioned Rubric object."""
    schema_version: int
    rubrics: List[RubricDict]

class Rubric:
    """Domain object describing a skill rubric."""

    def __init__(self, difficulty: str, explanations: List[str]) -> None:
        if False:
            while True:
                i = 10
        'Initializes a Rubric domain object.\n\n        Args:\n            difficulty: str. The question difficulty that this rubric addresses.\n            explanations: list(str). The different explanations for the\n                corresponding difficulty.\n        '
        self.difficulty = difficulty
        self.explanations = [html_cleaner.clean(explanation) for explanation in explanations]

    def to_dict(self) -> RubricDict:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict representing this Rubric domain object.\n\n        Returns:\n            dict. A dict, mapping all fields of Rubric instance.\n        '
        return {'difficulty': self.difficulty, 'explanations': self.explanations}

    @classmethod
    def from_dict(cls, rubric_dict: RubricDict) -> Rubric:
        if False:
            print('Hello World!')
        'Returns a Rubric domain object from a dict.\n\n        Args:\n            rubric_dict: dict. The dict representation of Rubric object.\n\n        Returns:\n            Rubric. The corresponding Rubric domain object.\n        '
        rubric = cls(rubric_dict['difficulty'], rubric_dict['explanations'])
        return rubric

    def validate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Validates various properties of the Rubric object.\n\n        Raises:\n            ValidationError. One or more attributes of the rubric are\n                invalid.\n        '
        if not isinstance(self.difficulty, str):
            raise utils.ValidationError('Expected difficulty to be a string, received %s' % self.difficulty)
        if self.difficulty not in constants.SKILL_DIFFICULTIES:
            raise utils.ValidationError('Invalid difficulty received for rubric: %s' % self.difficulty)
        if not isinstance(self.explanations, list):
            raise utils.ValidationError('Expected explanations to be a list, received %s' % self.explanations)
        for explanation in self.explanations:
            if not isinstance(explanation, str):
                raise utils.ValidationError('Expected each explanation to be a string, received %s' % explanation)
        if len(self.explanations) > 10:
            raise utils.ValidationError('Expected number of explanations to be less than or equal to 10, received %d' % len(self.explanations))
        for explanation in self.explanations:
            if len(explanation) > 300:
                raise utils.ValidationError('Explanation should be less than or equal to 300 chars, received %d chars' % len(explanation))
        if self.difficulty == constants.SKILL_DIFFICULTIES[1] and len(self.explanations) == 0:
            raise utils.ValidationError('Expected at least one explanation in medium level rubrics')

class WorkedExampleDict(TypedDict):
    """Dictionary representing the WorkedExample object."""
    question: state_domain.SubtitledHtmlDict
    explanation: state_domain.SubtitledHtmlDict

class WorkedExample:
    """Domain object for representing the worked_example dict."""

    def __init__(self, question: state_domain.SubtitledHtml, explanation: state_domain.SubtitledHtml) -> None:
        if False:
            i = 10
            return i + 15
        'Constructs a WorkedExample domain object.\n\n        Args:\n            question: SubtitledHtml. The example question.\n            explanation: SubtitledHtml. The explanation for the above example\n                question.\n        '
        self.question = question
        self.explanation = explanation

    def validate(self) -> None:
        if False:
            while True:
                i = 10
        'Validates various properties of the WorkedExample object.\n\n        Raises:\n            ValidationError. One or more attributes of the worked example are\n                invalid.\n        '
        if not isinstance(self.question, state_domain.SubtitledHtml):
            raise utils.ValidationError('Expected example question to be a SubtitledHtml object, received %s' % self.question)
        self.question.validate()
        if not isinstance(self.explanation, state_domain.SubtitledHtml):
            raise utils.ValidationError('Expected example explanation to be a SubtitledHtml object, received %s' % self.question)
        self.explanation.validate()

    def to_dict(self) -> WorkedExampleDict:
        if False:
            print('Hello World!')
        'Returns a dict representing this WorkedExample domain object.\n\n        Returns:\n            dict. A dict, mapping all fields of WorkedExample instance.\n        '
        return {'question': self.question.to_dict(), 'explanation': self.explanation.to_dict()}

    @classmethod
    def from_dict(cls, worked_example_dict: WorkedExampleDict) -> WorkedExample:
        if False:
            for i in range(10):
                print('nop')
        'Return a WorkedExample domain object from a dict.\n\n        Args:\n            worked_example_dict: dict. The dict representation of\n                WorkedExample object.\n\n        Returns:\n            WorkedExample. The corresponding WorkedExample domain object.\n        '
        worked_example = cls(state_domain.SubtitledHtml(worked_example_dict['question']['content_id'], worked_example_dict['question']['html']), state_domain.SubtitledHtml(worked_example_dict['explanation']['content_id'], worked_example_dict['explanation']['html']))
        return worked_example

class SkillContentsDict(TypedDict):
    """Dictionary representing the SkillContents object."""
    explanation: state_domain.SubtitledHtmlDict
    worked_examples: List[WorkedExampleDict]
    recorded_voiceovers: state_domain.RecordedVoiceoversDict
    written_translations: translation_domain.WrittenTranslationsDict

class VersionedSkillContentsDict(TypedDict):
    """Dictionary representing the versioned SkillContents object."""
    schema_version: int
    skill_contents: SkillContentsDict

class SkillContents:
    """Domain object representing the skill_contents dict."""

    def __init__(self, explanation: state_domain.SubtitledHtml, worked_examples: List[WorkedExample], recorded_voiceovers: state_domain.RecordedVoiceovers, written_translations: translation_domain.WrittenTranslations) -> None:
        if False:
            i = 10
            return i + 15
        'Constructs a SkillContents domain object.\n\n        Args:\n            explanation: SubtitledHtml. An explanation on how to apply the\n                skill.\n            worked_examples: list(WorkedExample). A list of worked examples\n                for the skill. Each element should be a WorkedExample object.\n            recorded_voiceovers: RecordedVoiceovers. The recorded voiceovers for\n                the skill contents and their translations in different\n                languages.\n            written_translations: WrittenTranslations. A text translation of\n                the skill contents.\n        '
        self.explanation = explanation
        self.worked_examples = worked_examples
        self.recorded_voiceovers = recorded_voiceovers
        self.written_translations = written_translations

    def validate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Validates various properties of the SkillContents object.\n\n        Raises:\n            ValidationError. One or more attributes of skill contents are\n                invalid.\n        '
        available_content_ids = set([])
        if not isinstance(self.explanation, state_domain.SubtitledHtml):
            raise utils.ValidationError('Expected skill explanation to be a SubtitledHtml object, received %s' % self.explanation)
        self.explanation.validate()
        available_content_ids.add(self.explanation.content_id)
        if not isinstance(self.worked_examples, list):
            raise utils.ValidationError('Expected worked examples to be a list, received %s' % self.worked_examples)
        for example in self.worked_examples:
            if not isinstance(example, WorkedExample):
                raise utils.ValidationError('Expected worked example to be a WorkedExample object, received %s' % example)
            example.validate()
            if example.question.content_id in available_content_ids:
                raise utils.ValidationError('Found a duplicate content id %s' % example.question.content_id)
            if example.explanation.content_id in available_content_ids:
                raise utils.ValidationError('Found a duplicate content id %s' % example.explanation.content_id)
            available_content_ids.add(example.question.content_id)
            available_content_ids.add(example.explanation.content_id)
        self.recorded_voiceovers.validate(list(available_content_ids))
        self.written_translations.validate(list(available_content_ids))

    def to_dict(self) -> SkillContentsDict:
        if False:
            while True:
                i = 10
        'Returns a dict representing this SkillContents domain object.\n\n        Returns:\n            dict. A dict, mapping all fields of SkillContents instance.\n        '
        return {'explanation': self.explanation.to_dict(), 'worked_examples': [worked_example.to_dict() for worked_example in self.worked_examples], 'recorded_voiceovers': self.recorded_voiceovers.to_dict(), 'written_translations': self.written_translations.to_dict()}

    @classmethod
    def from_dict(cls, skill_contents_dict: SkillContentsDict) -> SkillContents:
        if False:
            print('Hello World!')
        'Return a SkillContents domain object from a dict.\n\n        Args:\n            skill_contents_dict: dict. The dict representation of\n                SkillContents object.\n\n        Returns:\n            SkillContents. The corresponding SkillContents domain object.\n        '
        skill_contents = cls(state_domain.SubtitledHtml(skill_contents_dict['explanation']['content_id'], skill_contents_dict['explanation']['html']), [WorkedExample.from_dict(example) for example in skill_contents_dict['worked_examples']], state_domain.RecordedVoiceovers.from_dict(skill_contents_dict['recorded_voiceovers']), translation_domain.WrittenTranslations.from_dict(skill_contents_dict['written_translations']))
        return skill_contents

class SkillDict(TypedDict):
    """Dictionary representing the Skill object."""
    id: str
    description: str
    misconceptions: List[MisconceptionDict]
    rubrics: List[RubricDict]
    skill_contents: SkillContentsDict
    misconceptions_schema_version: int
    rubric_schema_version: int
    skill_contents_schema_version: int
    language_code: str
    version: int
    next_misconception_id: int
    superseding_skill_id: Optional[str]
    all_questions_merged: bool
    prerequisite_skill_ids: List[str]

class SerializableSkillDict(SkillDict):
    """Dictionary representing the serializable Skill object."""
    created_on: str
    last_updated: str

class Skill:
    """Domain object for an Oppia Skill."""

    def __init__(self, skill_id: str, description: str, misconceptions: List[Misconception], rubrics: List[Rubric], skill_contents: SkillContents, misconceptions_schema_version: int, rubric_schema_version: int, skill_contents_schema_version: int, language_code: str, version: int, next_misconception_id: int, superseding_skill_id: Optional[str], all_questions_merged: bool, prerequisite_skill_ids: List[str], created_on: Optional[datetime.datetime]=None, last_updated: Optional[datetime.datetime]=None) -> None:
        if False:
            return 10
        'Constructs a Skill domain object.\n\n        Args:\n            skill_id: str. The unique ID of the skill.\n            description: str. Describes the observable behaviour of the skill.\n            misconceptions: list(Misconception). The list of misconceptions\n                associated with the skill.\n            rubrics: list(Rubric). The list of rubrics that explain each\n                difficulty level of a skill.\n            skill_contents: SkillContents. The object representing the contents\n                of the skill.\n            misconceptions_schema_version: int. The schema version for the\n                misconceptions object.\n            rubric_schema_version: int. The schema version for the\n                rubric object.\n            skill_contents_schema_version: int. The schema version for the\n                skill_contents object.\n            language_code: str. The ISO 639-1 code for the language this\n                skill is written in.\n            version: int. The version of the skill.\n            next_misconception_id: int. The misconception id to be used by\n                the next misconception added.\n            superseding_skill_id: str|None. Skill ID of the skill we\n                merge this skill into. This is non null only if we indicate\n                that this skill is a duplicate and needs to be merged into\n                another one.\n            all_questions_merged: bool. Flag that indicates if all\n                questions are moved from this skill to the superseding skill.\n            prerequisite_skill_ids: list(str). The prerequisite skill IDs for\n                the skill.\n            created_on: datetime.datetime. Date and time when the skill is\n                created.\n            last_updated: datetime.datetime. Date and time when the\n                skill was last updated.\n        '
        self.id = skill_id
        self.description = description
        self.misconceptions = misconceptions
        self.skill_contents = skill_contents
        self.misconceptions_schema_version = misconceptions_schema_version
        self.rubric_schema_version = rubric_schema_version
        self.skill_contents_schema_version = skill_contents_schema_version
        self.language_code = language_code
        self.created_on = created_on
        self.last_updated = last_updated
        self.version = version
        self.rubrics = rubrics
        self.next_misconception_id = next_misconception_id
        self.superseding_skill_id = superseding_skill_id
        self.all_questions_merged = all_questions_merged
        self.prerequisite_skill_ids = prerequisite_skill_ids

    @classmethod
    def require_valid_skill_id(cls, skill_id: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Checks whether the skill id is a valid one.\n\n        Args:\n            skill_id: str. The skill id to validate.\n        '
        if not isinstance(skill_id, str):
            raise utils.ValidationError('Skill id should be a string.')
        if len(skill_id) != 12:
            raise utils.ValidationError('Invalid skill id.')

    @classmethod
    def require_valid_description(cls, description: str) -> None:
        if False:
            i = 10
            return i + 15
        'Checks whether the description of the skill is a valid one.\n\n        Args:\n            description: str. The description to validate.\n        '
        if not isinstance(description, str):
            raise utils.ValidationError('Description should be a string.')
        if description == '':
            raise utils.ValidationError('Description field should not be empty')
        description_length_limit = android_validation_constants.MAX_CHARS_IN_SKILL_DESCRIPTION
        if len(description) > description_length_limit:
            raise utils.ValidationError('Skill description should be less than %d chars, received %s' % (description_length_limit, description))

    def validate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Validates various properties of the Skill object.\n\n        Raises:\n            ValidationError. One or more attributes of skill are invalid.\n        '
        self.require_valid_description(self.description)
        Misconception.require_valid_misconception_id(self.next_misconception_id)
        if not isinstance(self.misconceptions_schema_version, int):
            raise utils.ValidationError('Expected misconceptions schema version to be an integer, received %s' % self.misconceptions_schema_version)
        if self.misconceptions_schema_version != feconf.CURRENT_MISCONCEPTIONS_SCHEMA_VERSION:
            raise utils.ValidationError('Expected misconceptions schema version to be %s, received %s' % (feconf.CURRENT_MISCONCEPTIONS_SCHEMA_VERSION, self.misconceptions_schema_version))
        if not isinstance(self.rubric_schema_version, int):
            raise utils.ValidationError('Expected rubric schema version to be an integer, received %s' % self.rubric_schema_version)
        if self.rubric_schema_version != feconf.CURRENT_RUBRIC_SCHEMA_VERSION:
            raise utils.ValidationError('Expected rubric schema version to be %s, received %s' % (feconf.CURRENT_RUBRIC_SCHEMA_VERSION, self.rubric_schema_version))
        if not isinstance(self.skill_contents_schema_version, int):
            raise utils.ValidationError('Expected skill contents schema version to be an integer, received %s' % self.skill_contents_schema_version)
        if self.skill_contents_schema_version != feconf.CURRENT_SKILL_CONTENTS_SCHEMA_VERSION:
            raise utils.ValidationError('Expected skill contents schema version to be %s, received %s' % (feconf.CURRENT_SKILL_CONTENTS_SCHEMA_VERSION, self.skill_contents_schema_version))
        if not isinstance(self.language_code, str):
            raise utils.ValidationError('Expected language code to be a string, received %s' % self.language_code)
        if not utils.is_valid_language_code(self.language_code):
            raise utils.ValidationError('Invalid language code: %s' % self.language_code)
        if not isinstance(self.skill_contents, SkillContents):
            raise utils.ValidationError('Expected skill_contents to be a SkillContents object, received %s' % self.skill_contents)
        self.skill_contents.validate()
        if not isinstance(self.rubrics, list):
            raise utils.ValidationError('Expected rubrics to be a list, received %s' % self.skill_contents)
        difficulties_list = []
        for rubric in self.rubrics:
            if not isinstance(rubric, Rubric):
                raise utils.ValidationError('Expected each rubric to be a Rubric object, received %s' % rubric)
            if rubric.difficulty in difficulties_list:
                raise utils.ValidationError('Duplicate rubric found for: %s' % rubric.difficulty)
            difficulties_list.append(rubric.difficulty)
            rubric.validate()
        if len(difficulties_list) != 3:
            raise utils.ValidationError('All 3 difficulties should be addressed in rubrics')
        if difficulties_list != constants.SKILL_DIFFICULTIES:
            raise utils.ValidationError('The difficulties should be ordered as follows [%s, %s, %s]' % (constants.SKILL_DIFFICULTIES[0], constants.SKILL_DIFFICULTIES[1], constants.SKILL_DIFFICULTIES[2]))
        if not isinstance(self.misconceptions, list):
            raise utils.ValidationError('Expected misconceptions to be a list, received %s' % self.misconceptions)
        if not isinstance(self.prerequisite_skill_ids, list):
            raise utils.ValidationError('Expected prerequisite_skill_ids to be a list, received %s' % self.prerequisite_skill_ids)
        for skill_id in self.prerequisite_skill_ids:
            if not isinstance(skill_id, str):
                raise utils.ValidationError('Expected each skill ID to be a string, received %s' % skill_id)
        misconception_id_list = []
        for misconception in self.misconceptions:
            if not isinstance(misconception, Misconception):
                raise utils.ValidationError('Expected each misconception to be a Misconception object, received %s' % misconception)
            if misconception.id in misconception_id_list:
                raise utils.ValidationError('Duplicate misconception ID found: %s' % misconception.id)
            misconception_id_list.append(misconception.id)
            if int(misconception.id) >= int(self.next_misconception_id):
                raise utils.ValidationError('The misconception with id %s is out of bounds.' % misconception.id)
            misconception.validate()
        if self.all_questions_merged and self.superseding_skill_id is None:
            raise utils.ValidationError('Expected a value for superseding_skill_id when all_questions_merged is True.')
        if self.superseding_skill_id is not None and self.all_questions_merged is None:
            raise utils.ValidationError('Expected a value for all_questions_merged when superseding_skill_id is set.')

    def to_dict(self) -> SkillDict:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict representing this Skill domain object.\n\n        Returns:\n            dict. A dict, mapping all fields of Skill instance.\n        '
        return {'id': self.id, 'description': self.description, 'misconceptions': [misconception.to_dict() for misconception in self.misconceptions], 'rubrics': [rubric.to_dict() for rubric in self.rubrics], 'skill_contents': self.skill_contents.to_dict(), 'language_code': self.language_code, 'misconceptions_schema_version': self.misconceptions_schema_version, 'rubric_schema_version': self.rubric_schema_version, 'skill_contents_schema_version': self.skill_contents_schema_version, 'version': self.version, 'next_misconception_id': self.next_misconception_id, 'superseding_skill_id': self.superseding_skill_id, 'all_questions_merged': self.all_questions_merged, 'prerequisite_skill_ids': self.prerequisite_skill_ids}

    def serialize(self) -> str:
        if False:
            print('Hello World!')
        'Returns the object serialized as a JSON string.\n\n        Returns:\n            str. JSON-encoded str encoding all of the information composing\n            the object.\n        '
        skill_dict: SerializableSkillDict = self.to_dict()
        skill_dict['version'] = self.version
        if self.created_on:
            skill_dict['created_on'] = utils.convert_naive_datetime_to_string(self.created_on)
        if self.last_updated:
            skill_dict['last_updated'] = utils.convert_naive_datetime_to_string(self.last_updated)
        return json.dumps(skill_dict)

    @classmethod
    def deserialize(cls, json_string: str) -> Skill:
        if False:
            print('Hello World!')
        'Returns a Skill domain object decoded from a JSON string.\n\n        Args:\n            json_string: str. A JSON-encoded string that can be\n                decoded into a dictionary representing a Skill.\n                Only call on strings that were created using serialize().\n\n        Returns:\n            Skill. The corresponding Skill domain object.\n        '
        skill_dict = json.loads(json_string)
        created_on = utils.convert_string_to_naive_datetime_object(skill_dict['created_on']) if 'created_on' in skill_dict else None
        last_updated = utils.convert_string_to_naive_datetime_object(skill_dict['last_updated']) if 'last_updated' in skill_dict else None
        skill = cls.from_dict(skill_dict, skill_version=skill_dict['version'], skill_created_on=created_on, skill_last_updated=last_updated)
        return skill

    @classmethod
    def from_dict(cls, skill_dict: SkillDict, skill_version: int=0, skill_created_on: Optional[datetime.datetime]=None, skill_last_updated: Optional[datetime.datetime]=None) -> Skill:
        if False:
            while True:
                i = 10
        'Returns a Skill domain object from a dict.\n\n        Args:\n            skill_dict: dict. The dictionary representation of skill\n                object.\n            skill_version: int. The version of the skill.\n            skill_created_on: datetime.datetime. Date and time when the\n                skill is created.\n            skill_last_updated: datetime.datetime. Date and time when the\n                skill was last updated.\n\n        Returns:\n            Skill. The corresponding Skill domain object.\n        '
        skill = cls(skill_dict['id'], skill_dict['description'], [Misconception.from_dict(misconception_dict) for misconception_dict in skill_dict['misconceptions']], [Rubric.from_dict(rubric_dict) for rubric_dict in skill_dict['rubrics']], SkillContents.from_dict(skill_dict['skill_contents']), skill_dict['misconceptions_schema_version'], skill_dict['rubric_schema_version'], skill_dict['skill_contents_schema_version'], skill_dict['language_code'], skill_version, skill_dict['next_misconception_id'], skill_dict['superseding_skill_id'], skill_dict['all_questions_merged'], skill_dict['prerequisite_skill_ids'], skill_created_on, skill_last_updated)
        return skill

    @classmethod
    def create_default_skill(cls, skill_id: str, description: str, rubrics: List[Rubric]) -> Skill:
        if False:
            while True:
                i = 10
        'Returns a skill domain object with default values. This is for\n        the frontend where a default blank skill would be shown to the user\n        when the skill is created for the first time.\n\n        Args:\n            skill_id: str. The unique id of the skill.\n            description: str. The initial description for the skill.\n            rubrics: list(Rubric). The list of rubrics for the skill.\n\n        Returns:\n            Skill. The Skill domain object with the default values.\n        '
        explanation_content_id = feconf.DEFAULT_SKILL_EXPLANATION_CONTENT_ID
        skill_contents = SkillContents(state_domain.SubtitledHtml(explanation_content_id, feconf.DEFAULT_SKILL_EXPLANATION), [], state_domain.RecordedVoiceovers.from_dict({'voiceovers_mapping': {explanation_content_id: {}}}), translation_domain.WrittenTranslations.from_dict({'translations_mapping': {explanation_content_id: {}}}))
        skill_contents.explanation.validate()
        return cls(skill_id, description, [], rubrics, skill_contents, feconf.CURRENT_MISCONCEPTIONS_SCHEMA_VERSION, feconf.CURRENT_RUBRIC_SCHEMA_VERSION, feconf.CURRENT_SKILL_CONTENTS_SCHEMA_VERSION, constants.DEFAULT_LANGUAGE_CODE, 0, 0, None, False, [])

    def generate_skill_misconception_id(self, misconception_id: int) -> str:
        if False:
            i = 10
            return i + 15
        "Given a misconception id, it returns the skill-misconception-id.\n        It is of the form <skill_id>-<misconception_id>.\n\n        Args:\n            misconception_id: int. The id of the misconception.\n\n        Returns:\n            str. The format is '<skill_id>-<misconception_id>', where skill_id\n            is the skill ID of the misconception and misconception_id is\n            the id of the misconception.\n        "
        return '%s-%d' % (self.id, misconception_id)

    @classmethod
    def convert_html_fields_in_skill_contents(cls, skill_contents_dict: SkillContentsDict, conversion_fn: Callable[[str], str]) -> SkillContentsDict:
        if False:
            return 10
        'Applies a conversion function on all the html strings in a skill\n        to migrate them to a desired state.\n\n        Args:\n            skill_contents_dict: dict. The dict representation of skill\n                contents.\n            conversion_fn: function. The conversion function to be applied on\n                the skill_contents_dict.\n\n        Returns:\n            dict. The converted skill_contents_dict.\n        '
        skill_contents_dict['explanation']['html'] = conversion_fn(skill_contents_dict['explanation']['html'])
        for (value_index, value) in enumerate(skill_contents_dict['worked_examples']):
            skill_contents_dict['worked_examples'][value_index]['question']['html'] = conversion_fn(value['question']['html'])
            skill_contents_dict['worked_examples'][value_index]['explanation']['html'] = conversion_fn(value['explanation']['html'])
        return skill_contents_dict

    @classmethod
    def _convert_skill_contents_v1_dict_to_v2_dict(cls, skill_contents_dict: SkillContentsDict) -> SkillContentsDict:
        if False:
            i = 10
            return i + 15
        'Converts v1 skill contents to the v2 schema. In the v2 schema,\n        the new Math components schema is introduced.\n\n        Args:\n            skill_contents_dict: dict. The v1 skill_contents_dict.\n\n        Returns:\n            dict. The converted skill_contents_dict.\n        '
        return cls.convert_html_fields_in_skill_contents(skill_contents_dict, html_validation_service.add_math_content_to_math_rte_components)

    @classmethod
    def _convert_skill_contents_v2_dict_to_v3_dict(cls, skill_contents_dict: SkillContentsDict) -> SkillContentsDict:
        if False:
            for i in range(10):
                print('nop')
        'Converts v2 skill contents to the v3 schema. The v3 schema\n        deprecates oppia-noninteractive-svgdiagram tag and converts existing\n        occurences of it to oppia-noninteractive-image tag.\n\n        Args:\n            skill_contents_dict: dict. The v1 skill_contents_dict.\n\n        Returns:\n            dict. The converted skill_contents_dict.\n        '
        return cls.convert_html_fields_in_skill_contents(skill_contents_dict, html_validation_service.convert_svg_diagram_tags_to_image_tags)

    @classmethod
    def _convert_skill_contents_v3_dict_to_v4_dict(cls, skill_contents_dict: SkillContentsDict) -> SkillContentsDict:
        if False:
            while True:
                i = 10
        'Converts v3 skill contents to the v4 schema. The v4 schema\n        fixes HTML encoding issues.\n\n        Args:\n            skill_contents_dict: dict. The v3 skill_contents_dict.\n\n        Returns:\n            dict. The converted skill_contents_dict.\n        '
        return cls.convert_html_fields_in_skill_contents(skill_contents_dict, html_validation_service.fix_incorrectly_encoded_chars)

    @classmethod
    def update_skill_contents_from_model(cls, versioned_skill_contents: VersionedSkillContentsDict, current_version: int) -> None:
        if False:
            return 10
        'Converts the skill_contents blob contained in the given\n        versioned_skill_contents dict from current_version to\n        current_version + 1. Note that the versioned_skill_contents being\n        passed in is modified in-place.\n\n        Args:\n            versioned_skill_contents: dict. A dict with two keys:\n                - schema_version: str. The schema version for the\n                    skill_contents dict.\n                - skill_contents: dict. The dict comprising the skill\n                    contents.\n            current_version: int. The current schema version of skill_contents.\n        '
        versioned_skill_contents['schema_version'] = current_version + 1
        conversion_fn = getattr(cls, '_convert_skill_contents_v%s_dict_to_v%s_dict' % (current_version, current_version + 1))
        versioned_skill_contents['skill_contents'] = conversion_fn(versioned_skill_contents['skill_contents'])

    @classmethod
    def update_misconceptions_from_model(cls, versioned_misconceptions: VersionedMisconceptionDict, current_version: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Converts the misconceptions blob contained in the given\n        versioned_misconceptions dict from current_version to\n        current_version + 1. Note that the versioned_misconceptions being\n        passed in is modified in-place.\n\n        Args:\n            versioned_misconceptions: dict. A dict with two keys:\n                - schema_version: str. The schema version for the\n                    misconceptions dict.\n                - misconceptions: list(dict). The list of dicts comprising the\n                    misconceptions of the skill.\n            current_version: int. The current schema version of misconceptions.\n        '
        versioned_misconceptions['schema_version'] = current_version + 1
        conversion_fn = getattr(cls, '_convert_misconception_v%s_dict_to_v%s_dict' % (current_version, current_version + 1))
        updated_misconceptions = []
        for misconception in versioned_misconceptions['misconceptions']:
            updated_misconceptions.append(conversion_fn(misconception))
        versioned_misconceptions['misconceptions'] = updated_misconceptions

    @classmethod
    def _convert_misconception_v1_dict_to_v2_dict(cls, misconception_dict: MisconceptionDict) -> MisconceptionDict:
        if False:
            for i in range(10):
                print('nop')
        'Converts v1 misconception schema to the v2 schema. In the v2 schema,\n        the field must_be_addressed has been added.\n\n        Args:\n            misconception_dict: dict. The v1 misconception dict.\n\n        Returns:\n            dict. The converted misconception_dict.\n        '
        misconception_dict['must_be_addressed'] = True
        return misconception_dict

    @classmethod
    def _convert_misconception_v2_dict_to_v3_dict(cls, misconception_dict: MisconceptionDict) -> MisconceptionDict:
        if False:
            print('Hello World!')
        'Converts v2 misconception schema to the v3 schema. In the v3 schema,\n        the new Math components schema is introduced.\n\n        Args:\n            misconception_dict: dict. The v2 misconception dict.\n\n        Returns:\n            dict. The converted misconception_dict.\n        '
        misconception_dict['notes'] = html_validation_service.add_math_content_to_math_rte_components(misconception_dict['notes'])
        misconception_dict['feedback'] = html_validation_service.add_math_content_to_math_rte_components(misconception_dict['feedback'])
        return misconception_dict

    @classmethod
    def _convert_misconception_v3_dict_to_v4_dict(cls, misconception_dict: MisconceptionDict) -> MisconceptionDict:
        if False:
            for i in range(10):
                print('nop')
        'Converts v3 misconception schema to the v4 schema. The v4 schema\n        deprecates oppia-noninteractive-svgdiagram tag and converts existing\n        occurences of it to oppia-noninteractive-image tag.\n\n        Args:\n            misconception_dict: dict. The v3 misconception dict.\n\n        Returns:\n            dict. The converted misconception_dict.\n        '
        misconception_dict['notes'] = html_validation_service.convert_svg_diagram_tags_to_image_tags(misconception_dict['notes'])
        misconception_dict['feedback'] = html_validation_service.convert_svg_diagram_tags_to_image_tags(misconception_dict['feedback'])
        return misconception_dict

    @classmethod
    def _convert_misconception_v4_dict_to_v5_dict(cls, misconception_dict: MisconceptionDict) -> MisconceptionDict:
        if False:
            print('Hello World!')
        'Converts v4 misconception schema to the v5 schema. The v5 schema\n        fixes HTML encoding issues.\n\n        Args:\n            misconception_dict: dict. The v4 misconception dict.\n\n        Returns:\n            dict. The converted misconception_dict.\n        '
        misconception_dict['notes'] = html_validation_service.fix_incorrectly_encoded_chars(misconception_dict['notes'])
        misconception_dict['feedback'] = html_validation_service.fix_incorrectly_encoded_chars(misconception_dict['feedback'])
        return misconception_dict

    @classmethod
    def _convert_rubric_v1_dict_to_v2_dict(cls, rubric_dict: RubricDict) -> RubricDict:
        if False:
            for i in range(10):
                print('nop')
        'Converts v1 rubric schema to the v2 schema. In the v2 schema,\n        multiple explanations have been added for each difficulty.\n\n        Args:\n            rubric_dict: dict. The v1 rubric dict.\n\n        Returns:\n            dict. The converted rubric_dict.\n        '
        explanation = rubric_dict['explanation']
        del rubric_dict['explanation']
        rubric_dict['explanations'] = [explanation]
        return rubric_dict

    @classmethod
    def _convert_rubric_v2_dict_to_v3_dict(cls, rubric_dict: RubricDict) -> RubricDict:
        if False:
            for i in range(10):
                print('nop')
        'Converts v2 rubric schema to the v3 schema. In the v3 schema,\n        the new Math components schema is introduced.\n\n        Args:\n            rubric_dict: dict. The v2 rubric dict.\n\n        Returns:\n            dict. The converted rubric_dict.\n        '
        for (explanation_index, explanation) in enumerate(rubric_dict['explanations']):
            rubric_dict['explanations'][explanation_index] = html_validation_service.add_math_content_to_math_rte_components(explanation)
        return rubric_dict

    @classmethod
    def _convert_rubric_v3_dict_to_v4_dict(cls, rubric_dict: RubricDict) -> RubricDict:
        if False:
            i = 10
            return i + 15
        'Converts v3 rubric schema to the v4 schema. The v4 schema\n        deprecates oppia-noninteractive-svgdiagram tag and converts existing\n        occurences of it to oppia-noninteractive-image tag.\n\n        Args:\n            rubric_dict: dict. The v2 rubric dict.\n\n        Returns:\n            dict. The converted rubric_dict.\n        '
        for (explanation_index, explanation) in enumerate(rubric_dict['explanations']):
            rubric_dict['explanations'][explanation_index] = html_validation_service.convert_svg_diagram_tags_to_image_tags(explanation)
        return rubric_dict

    @classmethod
    def _convert_rubric_v4_dict_to_v5_dict(cls, rubric_dict: RubricDict) -> RubricDict:
        if False:
            return 10
        'Converts v4 rubric schema to the v5 schema. The v4 schema\n        fixes HTML encoding issues.\n\n        Args:\n            rubric_dict: dict. The v4 rubric dict.\n\n        Returns:\n            dict. The converted rubric_dict.\n        '
        for (explanation_index, explanation) in enumerate(rubric_dict['explanations']):
            rubric_dict['explanations'][explanation_index] = html_validation_service.fix_incorrectly_encoded_chars(explanation)
        return rubric_dict

    @classmethod
    def update_rubrics_from_model(cls, versioned_rubrics: VersionedRubricDict, current_version: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Converts the rubrics blob contained in the given\n        versioned_rubrics dict from current_version to\n        current_version + 1. Note that the versioned_rubrics being\n        passed in is modified in-place.\n\n        Args:\n            versioned_rubrics: dict. A dict with two keys:\n                - schema_version: str. The schema version for the\n                    rubrics dict.\n                - rubrics: list(dict). The list of dicts comprising the\n                    rubrics of the skill.\n            current_version: int. The current schema version of rubrics.\n        '
        versioned_rubrics['schema_version'] = current_version + 1
        conversion_fn = getattr(cls, '_convert_rubric_v%s_dict_to_v%s_dict' % (current_version, current_version + 1))
        updated_rubrics = []
        for rubric in versioned_rubrics['rubrics']:
            updated_rubrics.append(conversion_fn(rubric))
        versioned_rubrics['rubrics'] = updated_rubrics

    def get_all_html_content_strings(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        'Returns all html strings that are part of the skill\n        (or any of its subcomponents).\n\n        Returns:\n            list(str). The list of html contents.\n        '
        html_content_strings = [self.skill_contents.explanation.html]
        for rubric in self.rubrics:
            for explanation in rubric.explanations:
                html_content_strings.append(explanation)
        for example in self.skill_contents.worked_examples:
            html_content_strings.append(example.question.html)
            html_content_strings.append(example.explanation.html)
        for misconception in self.misconceptions:
            html_content_strings.append(misconception.notes)
            html_content_strings.append(misconception.feedback)
        return html_content_strings

    def update_description(self, description: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Updates the description of the skill.\n\n        Args:\n            description: str. The new description of the skill.\n        '
        self.description = description

    def update_language_code(self, language_code: str) -> None:
        if False:
            while True:
                i = 10
        'Updates the language code of the skill.\n\n        Args:\n            language_code: str. The new language code of the skill.\n        '
        self.language_code = language_code

    def update_superseding_skill_id(self, superseding_skill_id: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Updates the superseding skill ID of the skill.\n\n        Args:\n            superseding_skill_id: str. ID of the skill that supersedes this one.\n        '
        self.superseding_skill_id = superseding_skill_id

    def record_that_all_questions_are_merged(self, all_questions_merged: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Updates the flag value which indicates if all questions are merged.\n\n        Args:\n            all_questions_merged: bool. Flag indicating if all questions are\n                merged to the superseding skill.\n        '
        self.all_questions_merged = all_questions_merged

    def update_explanation(self, explanation: state_domain.SubtitledHtml) -> None:
        if False:
            print('Hello World!')
        'Updates the explanation of the skill.\n\n        Args:\n            explanation: SubtitledHtml. The new explanation of the skill.\n        '
        old_content_ids = []
        if self.skill_contents.explanation:
            old_content_ids = [self.skill_contents.explanation.content_id]
        self.skill_contents.explanation = explanation
        new_content_ids = [self.skill_contents.explanation.content_id]
        self._update_content_ids_in_assets(old_content_ids, new_content_ids)

    def update_worked_examples(self, worked_examples: List[WorkedExample]) -> None:
        if False:
            while True:
                i = 10
        'Updates the worked examples list of the skill by performing a copy\n        of the provided list.\n\n        Args:\n            worked_examples: list(WorkedExample). The new worked examples of\n                the skill.\n        '
        old_content_ids = [example_field.content_id for example in self.skill_contents.worked_examples for example_field in (example.question, example.explanation)]
        self.skill_contents.worked_examples = list(worked_examples)
        new_content_ids = [example_field.content_id for example in self.skill_contents.worked_examples for example_field in (example.question, example.explanation)]
        self._update_content_ids_in_assets(old_content_ids, new_content_ids)

    def _update_content_ids_in_assets(self, old_ids_list: List[str], new_ids_list: List[str]) -> None:
        if False:
            i = 10
            return i + 15
        'Adds or deletes content ids in recorded_voiceovers and\n        written_translations.\n\n        Args:\n            old_ids_list: list(str). A list of content ids present earlier\n                in worked_examples.\n                state.\n            new_ids_list: list(str). A list of content ids currently present\n                in worked_examples.\n        '
        content_ids_to_delete = set(old_ids_list) - set(new_ids_list)
        content_ids_to_add = set(new_ids_list) - set(old_ids_list)
        written_translations = self.skill_contents.written_translations
        recorded_voiceovers = self.skill_contents.recorded_voiceovers
        for content_id in content_ids_to_delete:
            recorded_voiceovers.delete_content_id_for_voiceover(content_id)
            written_translations.delete_content_id_for_translation(content_id)
        for content_id in content_ids_to_add:
            recorded_voiceovers.add_content_id_for_voiceover(content_id)
            written_translations.add_content_id_for_translation(content_id)

    def _find_misconception_index(self, misconception_id: int) -> Optional[int]:
        if False:
            i = 10
            return i + 15
        'Returns the index of the misconception with the given misconception\n        id, or None if it is not in the misconceptions list.\n\n        Args:\n            misconception_id: int. The id of the misconception.\n\n        Returns:\n            int or None. The index of the corresponding misconception, or None\n            if there is no such misconception.\n        '
        for (ind, misconception) in enumerate(self.misconceptions):
            if misconception.id == misconception_id:
                return ind
        return None

    def add_misconception(self, misconception: Misconception) -> None:
        if False:
            while True:
                i = 10
        'Adds a new misconception to the skill.\n\n        Args:\n            misconception: Misconception. The misconception to be added.\n        '
        self.misconceptions.append(misconception)
        self.next_misconception_id = self.get_incremented_misconception_id(misconception.id)

    def _find_prerequisite_skill_id_index(self, skill_id_to_find: str) -> Optional[int]:
        if False:
            return 10
        'Returns the index of the skill_id in the prerequisite_skill_ids\n        array.\n\n        Args:\n            skill_id_to_find: str. The skill ID to search for.\n\n        Returns:\n            int|None. The index of the skill_id, if it exists or None.\n        '
        for (ind, skill_id) in enumerate(self.prerequisite_skill_ids):
            if skill_id == skill_id_to_find:
                return ind
        return None

    def add_prerequisite_skill(self, skill_id: str) -> None:
        if False:
            print('Hello World!')
        'Adds a prerequisite skill to the skill.\n\n        Args:\n            skill_id: str. The skill ID to add.\n\n        Raises:\n            ValueError. The skill is already a prerequisite skill.\n        '
        if self._find_prerequisite_skill_id_index(skill_id) is not None:
            raise ValueError('The skill is already a prerequisite skill.')
        self.prerequisite_skill_ids.append(skill_id)

    def delete_prerequisite_skill(self, skill_id: str) -> None:
        if False:
            while True:
                i = 10
        'Removes a prerequisite skill from the skill.\n\n        Args:\n            skill_id: str. The skill ID to remove.\n\n        Raises:\n            ValueError. The skill to remove is not a prerequisite skill.\n        '
        index = self._find_prerequisite_skill_id_index(skill_id)
        if index is None:
            raise ValueError('The skill to remove is not a prerequisite skill.')
        del self.prerequisite_skill_ids[index]

    def update_rubric(self, difficulty: str, explanations: List[str]) -> None:
        if False:
            return 10
        'Adds or updates the rubric of the given difficulty.\n\n        Args:\n            difficulty: str. The difficulty of the rubric.\n            explanations: list(str). The explanations for the rubric.\n\n        Raises:\n            ValueError. No rubric for given difficulty.\n        '
        for rubric in self.rubrics:
            if rubric.difficulty == difficulty:
                rubric.explanations = copy.deepcopy(explanations)
                return
        raise ValueError('There is no rubric for the given difficulty.')

    def get_incremented_misconception_id(self, misconception_id: int) -> int:
        if False:
            i = 10
            return i + 15
        'Returns the incremented misconception id.\n\n        Args:\n            misconception_id: int. The id of the misconception to be\n                incremented.\n\n        Returns:\n            int. The incremented misconception id.\n        '
        return misconception_id + 1

    def delete_misconception(self, misconception_id: int) -> None:
        if False:
            print('Hello World!')
        'Removes a misconception with the given id.\n\n        Args:\n            misconception_id: int. The id of the misconception to be removed.\n\n        Raises:\n            ValueError. There is no misconception with the given id.\n        '
        index = self._find_misconception_index(misconception_id)
        if index is None:
            raise ValueError('There is no misconception with the given id.')
        del self.misconceptions[index]

    def update_misconception_name(self, misconception_id: int, name: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Updates the name of the misconception with the given id.\n\n        Args:\n            misconception_id: int. The id of the misconception to be edited.\n            name: str. The new name of the misconception.\n\n        Raises:\n            ValueError. There is no misconception with the given id.\n        '
        index = self._find_misconception_index(misconception_id)
        if index is None:
            raise ValueError('There is no misconception with the given id.')
        self.misconceptions[index].name = name

    def update_misconception_must_be_addressed(self, misconception_id: int, must_be_addressed: bool) -> None:
        if False:
            i = 10
            return i + 15
        'Updates the must_be_addressed value of the misconception with the\n        given id.\n\n        Args:\n            misconception_id: int. The id of the misconception to be edited.\n            must_be_addressed: bool. The new must_be_addressed value for the\n                misconception.\n\n        Raises:\n            ValueError. There is no misconception with the given id.\n            ValueError. The must_be_addressed should be bool.\n        '
        if not isinstance(must_be_addressed, bool):
            raise ValueError('must_be_addressed should be a bool value.')
        index = self._find_misconception_index(misconception_id)
        if index is None:
            raise ValueError('There is no misconception with the given id.')
        self.misconceptions[index].must_be_addressed = must_be_addressed

    def update_misconception_notes(self, misconception_id: int, notes: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Updates the notes of the misconception with the given id.\n\n        Args:\n            misconception_id: int. The id of the misconception to be edited.\n            notes: str. The new notes of the misconception.\n\n        Raises:\n            ValueError. There is no misconception with the given id.\n        '
        index = self._find_misconception_index(misconception_id)
        if index is None:
            raise ValueError('There is no misconception with the given id.')
        self.misconceptions[index].notes = notes

    def update_misconception_feedback(self, misconception_id: int, feedback: str) -> None:
        if False:
            while True:
                i = 10
        'Updates the feedback of the misconception with the given id.\n\n        Args:\n            misconception_id: int. The id of the misconception to be edited.\n            feedback: str. The html string that corresponds to the new feedback\n                of the misconception.\n\n        Raises:\n            ValueError. There is no misconception with the given id.\n        '
        index = self._find_misconception_index(misconception_id)
        if index is None:
            raise ValueError('There is no misconception with the given id.')
        self.misconceptions[index].feedback = feedback

class SkillSummaryDict(TypedDict):
    """Dictionary representing the SkillSummary object."""
    id: str
    description: str
    language_code: str
    version: int
    misconception_count: int
    worked_examples_count: int
    skill_model_created_on: float
    skill_model_last_updated: float

class SkillSummary:
    """Domain object for Skill Summary."""

    def __init__(self, skill_id: str, description: str, language_code: str, version: int, misconception_count: int, worked_examples_count: int, skill_model_created_on: datetime.datetime, skill_model_last_updated: datetime.datetime) -> None:
        if False:
            i = 10
            return i + 15
        'Constructs a SkillSummary domain object.\n\n        Args:\n            skill_id: str. The unique id of the skill.\n            description: str. The short description of the skill.\n            language_code: str. The language code of the skill.\n            version: int. The version of the skill.\n            misconception_count: int. The number of misconceptions associated\n                with the skill.\n            worked_examples_count: int. The number of worked examples in the\n                skill.\n            skill_model_created_on: datetime.datetime. Date and time when\n                the skill model is created.\n            skill_model_last_updated: datetime.datetime. Date and time\n                when the skill model was last updated.\n        '
        self.id = skill_id
        self.description = description
        self.language_code = language_code
        self.version = version
        self.misconception_count = misconception_count
        self.worked_examples_count = worked_examples_count
        self.skill_model_created_on = skill_model_created_on
        self.skill_model_last_updated = skill_model_last_updated

    def validate(self) -> None:
        if False:
            i = 10
            return i + 15
        'Validates various properties of the Skill Summary object.\n\n        Raises:\n            ValidationError. One or more attributes of skill summary are\n                invalid.\n        '
        if not isinstance(self.description, str):
            raise utils.ValidationError('Description should be a string.')
        if self.description == '':
            raise utils.ValidationError('Description field should not be empty')
        if not isinstance(self.language_code, str):
            raise utils.ValidationError('Expected language code to be a string, received %s' % self.language_code)
        if not utils.is_valid_language_code(self.language_code):
            raise utils.ValidationError('Invalid language code: %s' % self.language_code)
        if not isinstance(self.misconception_count, int):
            raise utils.ValidationError("Expected misconception_count to be an int, received '%s'" % self.misconception_count)
        if self.misconception_count < 0:
            raise utils.ValidationError("Expected misconception_count to be non-negative, received '%s'" % self.misconception_count)
        if not isinstance(self.worked_examples_count, int):
            raise utils.ValidationError("Expected worked_examples_count to be an int, received '%s'" % self.worked_examples_count)
        if self.worked_examples_count < 0:
            raise utils.ValidationError("Expected worked_examples_count to be non-negative, received '%s'" % self.worked_examples_count)

    def to_dict(self) -> SkillSummaryDict:
        if False:
            while True:
                i = 10
        'Returns a dictionary representation of this domain object.\n\n        Returns:\n            dict. A dict representing this SkillSummary object.\n        '
        return {'id': self.id, 'description': self.description, 'language_code': self.language_code, 'version': self.version, 'misconception_count': self.misconception_count, 'worked_examples_count': self.worked_examples_count, 'skill_model_created_on': utils.get_time_in_millisecs(self.skill_model_created_on), 'skill_model_last_updated': utils.get_time_in_millisecs(self.skill_model_last_updated)}

class AugmentedSkillSummaryDict(TypedDict):
    """Dictionary representing the AugmentedSkillSummary object."""
    id: str
    description: str
    language_code: str
    version: int
    misconception_count: int
    worked_examples_count: int
    topic_names: List[str]
    classroom_names: List[str]
    skill_model_created_on: float
    skill_model_last_updated: float

class AugmentedSkillSummary:
    """Domain object for Augmented Skill Summary, which has all the properties
    of SkillSummary along with the topic names to which the skill is assigned
    and the classroom names to which the topics are assigned.
    """

    def __init__(self, skill_id: str, description: str, language_code: str, version: int, misconception_count: int, worked_examples_count: int, topic_names: List[str], classroom_names: List[str], skill_model_created_on: datetime.datetime, skill_model_last_updated: datetime.datetime) -> None:
        if False:
            print('Hello World!')
        'Constructs an AugmentedSkillSummary domain object.\n\n        Args:\n            skill_id: str. The unique id of the skill.\n            description: str. The short description of the skill.\n            language_code: str. The language code of the skill.\n            version: int. The version of the skill.\n            misconception_count: int. The number of misconceptions associated\n                with the skill.\n            worked_examples_count: int. The number of worked examples in the\n                skill.\n            topic_names: list(str). The names of the topics to which the skill\n                is assigned.\n            classroom_names: list(str). The names of the classrooms to which the\n                skill is assigned.\n            skill_model_created_on: datetime.datetime. Date and time when\n                the skill model is created.\n            skill_model_last_updated: datetime.datetime. Date and time\n                when the skill model was last updated.\n        '
        self.id = skill_id
        self.description = description
        self.language_code = language_code
        self.version = version
        self.misconception_count = misconception_count
        self.worked_examples_count = worked_examples_count
        self.skill_model_created_on = skill_model_created_on
        self.skill_model_last_updated = skill_model_last_updated
        self.topic_names = topic_names
        self.classroom_names = classroom_names

    def to_dict(self) -> AugmentedSkillSummaryDict:
        if False:
            print('Hello World!')
        'Returns a dictionary representation of this domain object.\n\n        Returns:\n            dict. A dict representing this AugmentedSkillSummary object.\n        '
        return {'id': self.id, 'description': self.description, 'language_code': self.language_code, 'version': self.version, 'misconception_count': self.misconception_count, 'worked_examples_count': self.worked_examples_count, 'topic_names': self.topic_names, 'classroom_names': self.classroom_names, 'skill_model_created_on': utils.get_time_in_millisecs(self.skill_model_created_on), 'skill_model_last_updated': utils.get_time_in_millisecs(self.skill_model_last_updated)}

class TopicAssignmentDict(TypedDict):
    """Dictionary representing the TopicAssignment object."""
    topic_id: str
    topic_name: str
    topic_version: int
    subtopic_id: Optional[int]

class TopicAssignment:
    """Domain object for Topic Assignment, which provides the details of a
    single topic (and, if applicable, the subtopic within that topic) to which
    the skill is assigned.
    """

    def __init__(self, topic_id: str, topic_name: str, topic_version: int, subtopic_id: Optional[int]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Constructs a TopicAssignment domain object.\n\n        Args:\n            topic_id: str. The unique id of the topic.\n            topic_name: str. The name of the topic.\n            topic_version: int. The current version of the topic to which the\n                skill is assigned.\n            subtopic_id: int or None. The id of the subtopic to which the skill\n                is assigned, or None if the skill is not assigned to any\n                subtopic.\n        '
        self.topic_id = topic_id
        self.topic_name = topic_name
        self.topic_version = topic_version
        self.subtopic_id = subtopic_id

    def to_dict(self) -> TopicAssignmentDict:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dictionary representation of this domain object.\n\n        Returns:\n            dict. A dict representing this TopicAssignment object.\n        '
        return {'topic_id': self.topic_id, 'topic_name': self.topic_name, 'topic_version': self.topic_version, 'subtopic_id': self.subtopic_id}

class UserSkillMasteryDict(TypedDict):
    """Dictionary representing the UserSkillMastery object."""
    user_id: str
    skill_id: str
    degree_of_mastery: float

class UserSkillMastery:
    """Domain object for a user's mastery of a particular skill."""

    def __init__(self, user_id: str, skill_id: str, degree_of_mastery: float) -> None:
        if False:
            i = 10
            return i + 15
        "Constructs a SkillMastery domain object for a user.\n\n        Args:\n            user_id: str. The user id of the user.\n            skill_id: str. The id of the skill.\n            degree_of_mastery: float. The user's mastery of the\n                corresponding skill.\n        "
        self.user_id = user_id
        self.skill_id = skill_id
        self.degree_of_mastery = degree_of_mastery

    def to_dict(self) -> UserSkillMasteryDict:
        if False:
            i = 10
            return i + 15
        'Returns a dictionary representation of this domain object.\n\n        Returns:\n            dict. A dict representing this SkillMastery object.\n        '
        return {'user_id': self.user_id, 'skill_id': self.skill_id, 'degree_of_mastery': self.degree_of_mastery}

    @classmethod
    def from_dict(cls, skill_mastery_dict: UserSkillMasteryDict) -> UserSkillMastery:
        if False:
            i = 10
            return i + 15
        'Returns a UserSkillMastery domain object from the given dict.\n\n        Args:\n            skill_mastery_dict: dict. A dict mapping all the fields of\n                UserSkillMastery object.\n\n        Returns:\n            SkillMastery. The SkillMastery domain object.\n        '
        return cls(skill_mastery_dict['user_id'], skill_mastery_dict['skill_id'], skill_mastery_dict['degree_of_mastery'])

class CategorizedSkills:
    """Domain object for representing categorized skills' ids and
    descriptions. Here, 'categorized skill' means that the skill is assigned
    to some topic. If a skill is assigned to a topic but not a
    subtopic, then it is termed as 'uncategorized' which also comes under
    CategorizedSkills because it is at least assigned to a topic.

    Attributes:
        categorized_skills: dict[str, dict[str, list(ShortSkillSummary)].
            The parent dict contains keys as topic names. The children dicts
            contain keys as subtopic titles and values as list of short skill
            summaries. An extra key called 'uncategorized' is present in every
            child dict to represent the skills that are not assigned to any
            subtopic but are assigned to the parent topic.
    """

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        'Constructs a CategorizedSkills domain object.'
        self.categorized_skills: Dict[str, Dict[str, List[ShortSkillSummary]]] = {}

    def add_topic(self, topic_name: str, subtopic_titles: List[str]) -> None:
        if False:
            i = 10
            return i + 15
        "Adds a topic to the categorized skills and initializes its\n        'uncategorized' and subtopic skills as empty lists.\n\n        Args:\n            topic_name: str. The name of the topic.\n            subtopic_titles: list(str). The list of subtopic titles of the\n                topic.\n\n        Raises:\n            ValidationError. Topic name is already added.\n        "
        if topic_name in self.categorized_skills:
            raise utils.ValidationError("Topic name '%s' is already added." % topic_name)
        self.categorized_skills[topic_name] = {}
        self.categorized_skills[topic_name]['uncategorized'] = []
        for subtopic_title in subtopic_titles:
            self.categorized_skills[topic_name][subtopic_title] = []

    def add_uncategorized_skill(self, topic_name: str, skill_id: str, skill_description: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Adds an uncategorized skill id and description for the given topic.\n\n        Args:\n            topic_name: str. The name of the topic.\n            skill_id: str. The id of the skill.\n            skill_description: str. The description of the skill.\n        '
        self.require_topic_name_to_be_added(topic_name)
        self.categorized_skills[topic_name]['uncategorized'].append(ShortSkillSummary(skill_id, skill_description))

    def add_subtopic_skill(self, topic_name: str, subtopic_title: str, skill_id: str, skill_description: str) -> None:
        if False:
            while True:
                i = 10
        'Adds a subtopic skill id and description for the given topic.\n\n        Args:\n            topic_name: str. The name of the topic.\n            subtopic_title: str. The title of the subtopic.\n            skill_id: str. The id of the skill.\n            skill_description: str. The description of the skill.\n        '
        self.require_topic_name_to_be_added(topic_name)
        self.require_subtopic_title_to_be_added(topic_name, subtopic_title)
        self.categorized_skills[topic_name][subtopic_title].append(ShortSkillSummary(skill_id, skill_description))

    def require_topic_name_to_be_added(self, topic_name: str) -> None:
        if False:
            return 10
        'Checks whether the given topic name is valid i.e. added to the\n        categorized skills dict.\n\n        Args:\n            topic_name: str. The name of the topic.\n\n        Raises:\n            ValidationError. Topic name is not added.\n        '
        if not topic_name in self.categorized_skills:
            raise utils.ValidationError("Topic name '%s' is not added." % topic_name)

    def require_subtopic_title_to_be_added(self, topic_name: str, subtopic_title: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Checks whether the given subtopic title is added to the\n        categorized skills dict under the given topic name.\n\n        Args:\n            topic_name: str. The name of the topic.\n            subtopic_title: str. The title of the subtopic.\n\n        Raises:\n            ValidationError. Subtopic title is not added.\n        '
        if not subtopic_title in self.categorized_skills[topic_name]:
            raise utils.ValidationError("Subtopic title '%s' is not added." % subtopic_title)

    def to_dict(self) -> Dict[str, Dict[str, List[ShortSkillSummaryDict]]]:
        if False:
            i = 10
            return i + 15
        'Returns a dictionary representation of this domain object.'
        categorized_skills_dict = copy.deepcopy(self.categorized_skills)
        result_categorized_skills_dict: Dict[str, Dict[str, List[ShortSkillSummaryDict]]] = {}
        for topic_name in categorized_skills_dict:
            result_categorized_skills_dict[topic_name] = {}
            for subtopic_title in categorized_skills_dict[topic_name]:
                result_categorized_skills_dict[topic_name][subtopic_title] = [short_skill_summary.to_dict() for short_skill_summary in categorized_skills_dict[topic_name][subtopic_title]]
        return result_categorized_skills_dict

class ShortSkillSummaryDict(TypedDict):
    """Dictionary representing the ShortSkillSummary object."""
    skill_id: str
    skill_description: str

class ShortSkillSummary:
    """Domain object for a short skill summary. It contains the id and
    description of the skill. It is different from the SkillSummary in the
    sense that the latter contains many other properties of the skill along with
    the skill id and description.
    """

    def __init__(self, skill_id: str, skill_description: str) -> None:
        if False:
            while True:
                i = 10
        'Constructs a ShortSkillSummary domain object.\n\n        Args:\n            skill_id: str. The id of the skill.\n            skill_description: str. The description of the skill.\n        '
        self.skill_id = skill_id
        self.skill_description = skill_description

    def to_dict(self) -> ShortSkillSummaryDict:
        if False:
            i = 10
            return i + 15
        'Returns a dictionary representation of this domain object.\n\n        Returns:\n            dict. A dict representing this ShortSkillSummary object.\n        '
        return {'skill_id': self.skill_id, 'skill_description': self.skill_description}

    @classmethod
    def from_skill_summary(cls, skill_summary: SkillSummary) -> ShortSkillSummary:
        if False:
            return 10
        'Returns a ShortSkillSummary domain object from the given skill\n        summary.\n\n        Args:\n            skill_summary: SkillSummary. The skill summary domain object.\n\n        Returns:\n            ShortSkillSummary. The ShortSkillSummary domain object.\n        '
        return cls(skill_summary.id, skill_summary.description)