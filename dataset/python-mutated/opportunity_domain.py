"""Domain object for contribution opportunities."""
from __future__ import annotations
from core import utils
from core.constants import constants
from typing import Dict, List, TypedDict

class PartialExplorationOpportunitySummaryDict(TypedDict):
    """A dictionary representing partial fields of
    ExplorationOpportunitySummary object.

    This dict has only required fields to represent
    an opportunity to a contributor.
    """
    id: str
    topic_name: str
    story_title: str
    chapter_title: str
    content_count: int
    translation_counts: Dict[str, int]
    translation_in_review_counts: Dict[str, int]
    is_pinned: bool

class ExplorationOpportunitySummaryDict(PartialExplorationOpportunitySummaryDict):
    """A dictionary representing ExplorationOpportunitySummary object.

    Contains all fields of an ExplorationOpportunitySummary object.
    It gets the required fields from PartialExplorationOpportunitySummaryDict.
    """
    topic_id: str
    story_id: str
    incomplete_translation_language_codes: List[str]
    language_codes_needing_voice_artists: List[str]
    language_codes_with_assigned_voice_artists: List[str]

class PinnedOpportunityDict(TypedDict):
    """A dictionary representing a PinnedOpportunity object."""
    language_code: str
    topic_id: str
    opportunity_id: str

class SkillOpportunityDict(TypedDict):
    """A dictionary representing SkillOpportunity object."""
    id: str
    skill_description: str
    question_count: int

class ExplorationOpportunitySummary:
    """The domain object for the translation and voiceover opportunities summary
    available in an exploration.
    """

    def __init__(self, exp_id: str, topic_id: str, topic_name: str, story_id: str, story_title: str, chapter_title: str, content_count: int, incomplete_translation_language_codes: List[str], translation_counts: Dict[str, int], language_codes_needing_voice_artists: List[str], language_codes_with_assigned_voice_artists: List[str], translation_in_review_counts: Dict[str, int], is_pinned: bool=False) -> None:
        if False:
            while True:
                i = 10
        'Constructs a ExplorationOpportunitySummary domain object.\n\n        Args:\n            exp_id: str. The unique id of the exploration.\n            topic_id: str. The unique id of the topic.\n            topic_name: str. The name of the topic.\n            story_id: str. The uniques id of the story.\n            story_title: str. The title of the story.\n            chapter_title: str. The title of the story chapter.\n            content_count: int. The total number of content available in the\n                exploration.\n            incomplete_translation_language_codes: list(str). A list of language\n                code in which the exploration translation is incomplete.\n            translation_counts: dict. A dict with language code as a key and\n                number of translation available in that language as the value.\n            language_codes_needing_voice_artists: list(str). A list of language\n                code in which the exploration needs voice artist.\n            language_codes_with_assigned_voice_artists: list(str). A list of\n                language code for which a voice-artist is already assigned to\n                the exploration.\n            translation_in_review_counts: dict. A dict with language code as a\n                key and number of translation in review in that language as the\n                value.\n            is_pinned: bool. Denotes whether the opportunity is pinned or not in\n                contributor dashboard.\n        '
        self.id = exp_id
        self.topic_id = topic_id
        self.topic_name = topic_name
        self.story_id = story_id
        self.story_title = story_title
        self.chapter_title = chapter_title
        self.content_count = content_count
        self.incomplete_translation_language_codes = incomplete_translation_language_codes
        self.translation_counts = translation_counts
        self.language_codes_needing_voice_artists = language_codes_needing_voice_artists
        self.language_codes_with_assigned_voice_artists = language_codes_with_assigned_voice_artists
        self.translation_in_review_counts = translation_in_review_counts
        self.is_pinned = is_pinned
        self.validate()

    @classmethod
    def from_dict(cls, exploration_opportunity_summary_dict: ExplorationOpportunitySummaryDict) -> 'ExplorationOpportunitySummary':
        if False:
            i = 10
            return i + 15
        'Return a ExplorationOpportunitySummary domain object from a dict.\n\n        Args:\n            exploration_opportunity_summary_dict: dict. The dict representation\n                of ExplorationOpportunitySummary object.\n\n        Returns:\n            ExplorationOpportunitySummary. The corresponding\n            ExplorationOpportunitySummary domain object.\n        '
        return cls(exploration_opportunity_summary_dict['id'], exploration_opportunity_summary_dict['topic_id'], exploration_opportunity_summary_dict['topic_name'], exploration_opportunity_summary_dict['story_id'], exploration_opportunity_summary_dict['story_title'], exploration_opportunity_summary_dict['chapter_title'], exploration_opportunity_summary_dict['content_count'], exploration_opportunity_summary_dict['incomplete_translation_language_codes'], exploration_opportunity_summary_dict['translation_counts'], exploration_opportunity_summary_dict['language_codes_needing_voice_artists'], exploration_opportunity_summary_dict['language_codes_with_assigned_voice_artists'], exploration_opportunity_summary_dict['translation_in_review_counts'])

    def to_dict(self) -> PartialExplorationOpportunitySummaryDict:
        if False:
            print('Hello World!')
        'Return a copy of the object as a dictionary. It includes all\n        necessary information to represent an opportunity.\n\n        NOTE: The returned dict has only those data which are required to\n        represent the opportunity to a contributor.\n\n        Returns:\n            dict. A dict mapping the fields of ExplorationOpportunitySummary\n            instance which are required to represent the opportunity to a\n            contributor.\n        '
        return {'id': self.id, 'topic_name': self.topic_name, 'story_title': self.story_title, 'chapter_title': self.chapter_title, 'content_count': self.content_count, 'translation_counts': self.translation_counts, 'translation_in_review_counts': self.translation_in_review_counts, 'is_pinned': self.is_pinned}

    def validate(self) -> None:
        if False:
            print('Hello World!')
        'Validates various properties of the object.\n\n        Raises:\n            ValidationError. One or more attributes of the object are invalid.\n        '
        if self.content_count < 0:
            raise utils.ValidationError('Expected content_count to be a non-negative integer, received %s' % self.content_count)
        allowed_language_codes = [language['id'] for language in constants.SUPPORTED_AUDIO_LANGUAGES]
        if not set(self.language_codes_with_assigned_voice_artists).isdisjoint(self.language_codes_needing_voice_artists):
            raise utils.ValidationError('Expected voice_artist "needed" and "assigned" list of languages to be disjoint, received: %s, %s' % (self.language_codes_needing_voice_artists, self.language_codes_with_assigned_voice_artists))
        self._validate_translation_counts(self.translation_counts)
        self._validate_translation_counts(self.translation_in_review_counts)
        expected_set_of_all_languages = set(self.incomplete_translation_language_codes + self.language_codes_needing_voice_artists + self.language_codes_with_assigned_voice_artists)
        for language_code in expected_set_of_all_languages:
            if language_code not in allowed_language_codes:
                raise utils.ValidationError('Invalid language_code: %s' % language_code)
        if expected_set_of_all_languages != set(allowed_language_codes):
            raise utils.ValidationError('Expected set of all languages available in incomplete_translation, needs_voiceover and assigned_voiceover to be the same as the supported audio languages, received %s' % list(sorted(expected_set_of_all_languages)))

    def _validate_translation_counts(self, translation_counts: Dict[str, int]) -> None:
        if False:
            while True:
                i = 10
        'Validates per-language counts of translations.\n\n        Args:\n            translation_counts: dict. A dict with language code as a key and\n                number of translations in that language as the value.\n\n        Raises:\n            ValidationError. One or more attributes of the object are invalid.\n        '
        for (language_code, count) in translation_counts.items():
            if not utils.is_supported_audio_language_code(language_code):
                raise utils.ValidationError('Invalid language_code: %s' % language_code)
            if count < 0:
                raise utils.ValidationError('Expected count for language_code %s to be a non-negative integer, received %s' % (language_code, count))
            if count > self.content_count:
                raise utils.ValidationError('Expected translation count for language_code %s to be less than or equal to content_count(%s), received %s' % (language_code, self.content_count, count))

class SkillOpportunity:
    """The domain object for skill opportunities."""

    def __init__(self, skill_id: str, skill_description: str, question_count: int) -> None:
        if False:
            return 10
        'Constructs a SkillOpportunity domain object.\n\n        Args:\n            skill_id: str. The unique id of the skill.\n            skill_description: str. The title of the skill.\n            question_count: int. The total number of questions for the skill.\n        '
        self.id = skill_id
        self.skill_description = skill_description
        self.question_count = question_count
        self.validate()

    def validate(self) -> None:
        if False:
            return 10
        'Validates various properties of the object.\n\n        Raises:\n            ValidationError. One or more attributes of the object are invalid.\n        '
        if self.question_count < 0:
            raise utils.ValidationError('Expected question_count to be a non-negative integer, received %s' % self.question_count)

    @classmethod
    def from_dict(cls, skill_opportunity_dict: SkillOpportunityDict) -> 'SkillOpportunity':
        if False:
            print('Hello World!')
        'Return a SkillOpportunity domain object from a dict.\n\n        Args:\n            skill_opportunity_dict: dict. The dict representation of a\n                SkillOpportunity object.\n\n        Returns:\n            SkillOpportunity. The corresponding SkillOpportunity domain object.\n        '
        return cls(skill_opportunity_dict['id'], skill_opportunity_dict['skill_description'], skill_opportunity_dict['question_count'])

    def to_dict(self) -> SkillOpportunityDict:
        if False:
            print('Hello World!')
        'Returns a copy of the object as a dictionary. It includes all\n        necessary information to represent an opportunity.\n\n        Returns:\n            dict. A dict mapping the fields of SkillOpportunity instance which\n            are required to represent the opportunity to a contributor.\n        '
        return {'id': self.id, 'skill_description': self.skill_description, 'question_count': self.question_count}

class PinnedOpportunity:
    """The domain object for pinned translation opportunities in
    the contributor dashboard.
    """

    def __init__(self, language_code: str, topic_id: str, opportunity_id: str) -> None:
        if False:
            print('Hello World!')
        'Constructs a PinnedOpportunity domain object.\n\n        Args:\n            language_code: str. The ISO 639-1 language code for which the\n                opportunity is pinned.\n            topic_id: str. The ID of the topic for which the\n                opportunity is pinned.\n            opportunity_id: str. The ID of the pinned opportunity.\n        '
        self.language_code = language_code
        self.topic_id = topic_id
        self.opportunity_id = opportunity_id

    @classmethod
    def from_dict(cls, pinned_opportunity_dict: PinnedOpportunityDict) -> 'PinnedOpportunity':
        if False:
            print('Hello World!')
        'Returns a PinnedOpportunity domain object from a dict.\n\n        Args:\n            pinned_opportunity_dict: dict. The dict representation of a\n                PinnedOpportunity object.\n\n        Returns:\n            PinnedOpportunity. The corresponding PinnedOpportunity\n            domain object.\n        '
        return cls(pinned_opportunity_dict['language_code'], pinned_opportunity_dict['topic_id'], pinned_opportunity_dict['opportunity_id'])

    def to_dict(self) -> PinnedOpportunityDict:
        if False:
            print('Hello World!')
        'Returns a copy of the object as a dictionary. It includes all\n        necessary information to represent a pinned opportunity.\n\n        Returns:\n            dict. A dict mapping the fields of PinnedOpportunity instance.\n        '
        return {'language_code': self.language_code, 'topic_id': self.topic_id, 'opportunity_id': self.opportunity_id}