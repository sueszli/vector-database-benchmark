"""Registry for Oppia suggestions. Contains a BaseSuggestion class and
subclasses for each type of suggestion.
"""
from __future__ import annotations
import copy
import datetime
from core import feconf
from core import utils
from core.constants import constants
from core.domain import change_domain
from core.domain import exp_domain
from core.domain import exp_fetchers
from core.domain import exp_services
from core.domain import fs_services
from core.domain import html_cleaner
from core.domain import opportunity_services
from core.domain import platform_feature_services
from core.domain import platform_parameter_list
from core.domain import question_domain
from core.domain import question_services
from core.domain import skill_domain
from core.domain import skill_fetchers
from core.domain import state_domain
from core.domain import topic_fetchers
from core.domain import translation_domain
from core.domain import translation_services
from core.domain import user_services
from core.platform import models
from extensions import domain
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Type, TypedDict, Union, cast
MYPY = False
if MYPY:
    from mypy_imports import suggestion_models
(suggestion_models,) = models.Registry.import_models([models.Names.SUGGESTION])

class BaseSuggestionDict(TypedDict):
    """Dictionary representing the BaseSuggestion object."""
    suggestion_id: str
    suggestion_type: str
    target_type: str
    target_id: str
    target_version_at_submission: int
    status: str
    author_name: str
    final_reviewer_id: Optional[str]
    change: Dict[str, change_domain.AcceptableChangeDictTypes]
    score_category: str
    language_code: str
    last_updated: float
    created_on: float
    edited_by_reviewer: bool

class BaseSuggestion:
    """Base class for a suggestion.

    Attributes:
        suggestion_id: str. The ID of the suggestion.
        suggestion_type: str. The type of the suggestion.
        target_type: str. The type of target entity being edited.
        target_id: str. The ID of the target entity being edited.
        target_version_at_submission: int. The version number of the target
            entity at the time of creation of the suggestion.
        status: str. The status of the suggestion.
        author_id: str. The ID of the user who submitted the suggestion.
        final_reviewer_id: str. The ID of the reviewer who has accepted/rejected
            the suggestion.
        change: Change. The details of the suggestion. This should be an
            object of type ExplorationChange, TopicChange, etc.
        score_category: str. The scoring category for the suggestion.
        last_updated: datetime.datetime. Date and time when the suggestion
            was last updated.
        language_code: str|None. The ISO 639-1 code used to query suggestions
            by language, or None if the suggestion type is not queryable by
            language.
        edited_by_reviewer: bool. Whether the suggestion is edited by the
            reviewer.
    """
    suggestion_id: str
    suggestion_type: str
    target_type: str
    target_id: str
    target_version_at_submission: int
    author_id: str
    change: change_domain.BaseChange
    score_category: str
    last_updated: datetime.datetime
    created_on: datetime.datetime
    language_code: str
    edited_by_reviewer: bool
    image_context: str

    def __init__(self, status: str, final_reviewer_id: Optional[str]) -> None:
        if False:
            return 10
        'Initializes a Suggestion object.'
        self.status = status
        self.final_reviewer_id = final_reviewer_id

    def to_dict(self) -> BaseSuggestionDict:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict representation of a suggestion object.\n\n        Returns:\n            dict. A dict representation of a suggestion object.\n        '
        return {'suggestion_id': self.suggestion_id, 'suggestion_type': self.suggestion_type, 'target_type': self.target_type, 'target_id': self.target_id, 'target_version_at_submission': self.target_version_at_submission, 'status': self.status, 'author_name': self.get_author_name(), 'final_reviewer_id': self.final_reviewer_id, 'change': self.change.to_dict(), 'score_category': self.score_category, 'language_code': self.language_code, 'last_updated': utils.get_time_in_millisecs(self.last_updated), 'created_on': utils.get_time_in_millisecs(self.created_on), 'edited_by_reviewer': self.edited_by_reviewer}

    def get_score_type(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Returns the first part of the score category. The first part refers\n        to the the type of scoring. The value of this part will be among\n        suggestion_models.SCORE_TYPE_CHOICES.\n\n        Returns:\n            str. The first part of the score category.\n        '
        return self.score_category.split(suggestion_models.SCORE_CATEGORY_DELIMITER)[0]

    def get_author_name(self) -> str:
        if False:
            while True:
                i = 10
        "Returns the author's username.\n\n        Returns:\n            str. The username of the author of the suggestion.\n        "
        return user_services.get_username(self.author_id)

    def get_score_sub_type(self) -> str:
        if False:
            print('Hello World!')
        'Returns the second part of the score category. The second part refers\n        to the specific area where the author needs to be scored. This can be\n        the category of the exploration, the language of the suggestion, or the\n        skill linked to the question.\n\n        Returns:\n            str. The second part of the score category.\n        '
        return self.score_category.split(suggestion_models.SCORE_CATEGORY_DELIMITER)[1]

    def set_suggestion_status_to_accepted(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Sets the status of the suggestion to accepted.'
        self.status = suggestion_models.STATUS_ACCEPTED

    def set_suggestion_status_to_in_review(self) -> None:
        if False:
            print('Hello World!')
        'Sets the status of the suggestion to in review.'
        self.status = suggestion_models.STATUS_IN_REVIEW

    def set_suggestion_status_to_rejected(self) -> None:
        if False:
            print('Hello World!')
        'Sets the status of the suggestion to rejected.'
        self.status = suggestion_models.STATUS_REJECTED

    def set_final_reviewer_id(self, reviewer_id: str) -> None:
        if False:
            i = 10
            return i + 15
        'Sets the final reviewer id of the suggestion to be reviewer_id.\n\n        Args:\n            reviewer_id: str. The ID of the user who completed the review.\n        '
        self.final_reviewer_id = reviewer_id

    def validate(self) -> None:
        if False:
            print('Hello World!')
        'Validates the BaseSuggestion object. Each subclass must implement\n        this function.\n\n        The subclasses must validate the change and score_category fields.\n\n        Raises:\n            ValidationError. One or more attributes of the BaseSuggestion object\n                are invalid.\n        '
        if self.suggestion_type not in feconf.SUGGESTION_TYPE_CHOICES:
            raise utils.ValidationError('Expected suggestion_type to be among allowed choices, received %s' % self.suggestion_type)
        if self.target_type not in feconf.SUGGESTION_TARGET_TYPE_CHOICES:
            raise utils.ValidationError('Expected target_type to be among allowed choices, received %s' % self.target_type)
        if not isinstance(self.target_id, str):
            raise utils.ValidationError('Expected target_id to be a string, received %s' % type(self.target_id))
        if not isinstance(self.target_version_at_submission, int):
            raise utils.ValidationError('Expected target_version_at_submission to be an int, received %s' % type(self.target_version_at_submission))
        if self.status not in suggestion_models.STATUS_CHOICES:
            raise utils.ValidationError('Expected status to be among allowed choices, received %s' % self.status)
        if not isinstance(self.author_id, str):
            raise utils.ValidationError('Expected author_id to be a string, received %s' % type(self.author_id))
        if not utils.is_user_id_valid(self.author_id, allow_pseudonymous_id=True):
            raise utils.ValidationError('Expected author_id to be in a valid user ID format, received %s' % self.author_id)
        if self.final_reviewer_id is not None:
            if not isinstance(self.final_reviewer_id, str):
                raise utils.ValidationError('Expected final_reviewer_id to be a string, received %s' % type(self.final_reviewer_id))
            if not utils.is_user_id_valid(self.final_reviewer_id, allow_system_user_id=True, allow_pseudonymous_id=True):
                raise utils.ValidationError('Expected final_reviewer_id to be in a valid user ID format, received %s' % self.final_reviewer_id)
        if not isinstance(self.score_category, str):
            raise utils.ValidationError('Expected score_category to be a string, received %s' % type(self.score_category))
        if suggestion_models.SCORE_CATEGORY_DELIMITER not in self.score_category:
            raise utils.ValidationError('Expected score_category to be of the form score_type%sscore_sub_type, received %s' % (suggestion_models.SCORE_CATEGORY_DELIMITER, self.score_category))
        if len(self.score_category.split(suggestion_models.SCORE_CATEGORY_DELIMITER)) != 2:
            raise utils.ValidationError('Expected score_category to be of the form score_type%sscore_sub_type, received %s' % (suggestion_models.SCORE_CATEGORY_DELIMITER, self.score_category))
        if self.get_score_type() not in suggestion_models.SCORE_TYPE_CHOICES:
            raise utils.ValidationError('Expected the first part of score_category to be among allowed choices, received %s' % self.get_score_type())

    def accept(self, commit_msg: str) -> None:
        if False:
            i = 10
            return i + 15
        'Accepts the suggestion. Each subclass must implement this\n        function.\n        '
        raise NotImplementedError('Subclasses of BaseSuggestion should implement accept.')

    def pre_accept_validate(self) -> None:
        if False:
            print('Hello World!')
        'Performs referential validation. This function needs to be called\n        before accepting the suggestion.\n        '
        raise NotImplementedError('Subclasses of BaseSuggestion should implement pre_accept_validate.')

    def populate_old_value_of_change(self) -> None:
        if False:
            while True:
                i = 10
        'Populates the old_value field of the change.'
        raise NotImplementedError('Subclasses of BaseSuggestion should implement populate_old_value_of_change.')

    def pre_update_validate(self, change: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Performs the pre update validation. This function needs to be called\n        before updating the suggestion.\n        '
        raise NotImplementedError('Subclasses of BaseSuggestion should implement pre_update_validate.')

    def get_all_html_content_strings(self) -> List[str]:
        if False:
            print('Hello World!')
        'Gets all html content strings used in this suggestion.'
        raise NotImplementedError('Subclasses of BaseSuggestion should implement get_all_html_content_strings.')

    def get_target_entity_html_strings(self) -> List[str]:
        if False:
            return 10
        'Gets all html content strings from target entity used in the\n        suggestion.\n        '
        raise NotImplementedError('Subclasses of BaseSuggestion should implement get_target_entity_html_strings.')

    def get_new_image_filenames_added_in_suggestion(self) -> List[str]:
        if False:
            return 10
        'Returns the list of newly added image filenames in the suggestion.\n\n        Returns:\n            list(str). A list of newly added image filenames in the suggestion.\n        '
        html_list = self.get_all_html_content_strings()
        all_image_filenames = html_cleaner.get_image_filenames_from_html_strings(html_list)
        target_entity_html_list = self.get_target_entity_html_strings()
        target_image_filenames = html_cleaner.get_image_filenames_from_html_strings(target_entity_html_list)
        new_image_filenames = utils.compute_list_difference(all_image_filenames, target_image_filenames)
        return new_image_filenames

    def _copy_new_images_to_target_entity_storage(self) -> None:
        if False:
            print('Hello World!')
        'Copy newly added images in suggestion to the target entity\n        storage.\n        '
        new_image_filenames = self.get_new_image_filenames_added_in_suggestion()
        fs_services.copy_images(self.image_context, self.target_id, self.target_type, self.target_id, new_image_filenames)

    def convert_html_in_suggestion_change(self, conversion_fn: Callable[[str], str]) -> None:
        if False:
            i = 10
            return i + 15
        'Checks for HTML fields in a suggestion change and converts it\n        according to the conversion function.\n        '
        raise NotImplementedError('Subclasses of BaseSuggestion should implement convert_html_in_suggestion_change.')

    @property
    def is_handled(self) -> bool:
        if False:
            print('Hello World!')
        'Returns if the suggestion has either been accepted or rejected.\n\n        Returns:\n            bool. Whether the suggestion has been handled or not.\n        '
        return self.status != suggestion_models.STATUS_IN_REVIEW

class SuggestionEditStateContent(BaseSuggestion):
    """Domain object for a suggestion of type
    SUGGESTION_TYPE_EDIT_STATE_CONTENT.
    """

    def __init__(self, suggestion_id: str, target_id: str, target_version_at_submission: int, status: str, author_id: str, final_reviewer_id: Optional[str], change: Mapping[str, change_domain.AcceptableChangeDictTypes], score_category: str, language_code: Optional[str], edited_by_reviewer: bool, last_updated: Optional[datetime.datetime]=None, created_on: Optional[datetime.datetime]=None) -> None:
        if False:
            return 10
        'Initializes an object of type SuggestionEditStateContent\n        corresponding to the SUGGESTION_TYPE_EDIT_STATE_CONTENT choice.\n        '
        super().__init__(status, final_reviewer_id)
        self.suggestion_id = suggestion_id
        self.suggestion_type = feconf.SUGGESTION_TYPE_EDIT_STATE_CONTENT
        self.target_type = feconf.ENTITY_TYPE_EXPLORATION
        self.target_id = target_id
        self.target_version_at_submission = target_version_at_submission
        self.author_id = author_id
        self.change: exp_domain.EditExpStatePropertyContentCmd = exp_domain.EditExpStatePropertyContentCmd(change)
        self.score_category = score_category
        self.language_code = language_code
        self.last_updated = last_updated
        self.created_on = created_on
        self.edited_by_reviewer = edited_by_reviewer
        self.image_context = None

    def validate(self) -> None:
        if False:
            return 10
        'Validates a suggestion object of type SuggestionEditStateContent.\n\n        Raises:\n            ValidationError. One or more attributes of the\n                SuggestionEditStateContent object are invalid.\n        '
        super().validate()
        if not isinstance(self.change, exp_domain.ExplorationChange):
            raise utils.ValidationError('Expected change to be an ExplorationChange, received %s' % type(self.change))
        if self.get_score_type() != suggestion_models.SCORE_TYPE_CONTENT:
            raise utils.ValidationError('Expected the first part of score_category to be %s , received %s' % (suggestion_models.SCORE_TYPE_CONTENT, self.get_score_type()))
        if self.change.cmd != exp_domain.CMD_EDIT_STATE_PROPERTY:
            raise utils.ValidationError('Expected cmd to be %s, received %s' % (exp_domain.CMD_EDIT_STATE_PROPERTY, self.change.cmd))
        if self.change.property_name != exp_domain.STATE_PROPERTY_CONTENT:
            raise utils.ValidationError('Expected property_name to be %s, received %s' % (exp_domain.STATE_PROPERTY_CONTENT, self.change.property_name))
        if self.language_code is not None:
            raise utils.ValidationError('Expected language_code to be None, received %s' % self.language_code)

    def pre_accept_validate(self) -> None:
        if False:
            i = 10
            return i + 15
        'Performs referential validation. This function needs to be called\n        before accepting the suggestion.\n        '
        self.validate()
        states = exp_fetchers.get_exploration_by_id(self.target_id).states
        if self.change.state_name not in states:
            raise utils.ValidationError('Expected %s to be a valid state name' % self.change.state_name)

    def _get_change_list_for_accepting_edit_state_content_suggestion(self) -> List[exp_domain.ExplorationChange]:
        if False:
            while True:
                i = 10
        'Gets a complete change for the SuggestionEditStateContent.\n\n        Returns:\n            list(ExplorationChange). The change_list corresponding to the\n            suggestion.\n        '
        change = self.change
        exploration = exp_fetchers.get_exploration_by_id(self.target_id)
        old_content = exploration.states[self.change.state_name].content.to_dict()
        change.old_value = old_content
        change.new_value['content_id'] = old_content['content_id']
        return [change]

    def populate_old_value_of_change(self) -> None:
        if False:
            return 10
        'Populates old value of the change.'
        exploration = exp_fetchers.get_exploration_by_id(self.target_id)
        if self.change.state_name not in exploration.states:
            old_content = None
        else:
            old_content = exploration.states[self.change.state_name].content.to_dict()
        self.change.old_value = old_content

    def accept(self, commit_message: str) -> None:
        if False:
            i = 10
            return i + 15
        'Accepts the suggestion.\n\n        Args:\n            commit_message: str. The commit message.\n        '
        change_list = self._get_change_list_for_accepting_edit_state_content_suggestion()
        assert self.final_reviewer_id is not None
        exp_services.update_exploration(self.final_reviewer_id, self.target_id, change_list, commit_message)

    def pre_update_validate(self, change: exp_domain.EditExpStatePropertyContentCmd) -> None:
        if False:
            i = 10
            return i + 15
        'Performs the pre update validation. This function needs to be called\n        before updating the suggestion.\n\n        Args:\n            change: ExplorationChange. The new change.\n\n        Raises:\n            ValidationError. Invalid new change.\n        '
        if self.change.cmd != change.cmd:
            raise utils.ValidationError('The new change cmd must be equal to %s' % self.change.cmd)
        if self.change.property_name != change.property_name:
            raise utils.ValidationError('The new change property_name must be equal to %s' % self.change.property_name)
        if self.change.state_name != change.state_name:
            raise utils.ValidationError('The new change state_name must be equal to %s' % self.change.state_name)
        if self.change.new_value['html'] == change.new_value['html']:
            raise utils.ValidationError('The new html must not match the old html')

    def get_all_html_content_strings(self) -> List[str]:
        if False:
            while True:
                i = 10
        'Gets all html content strings used in this suggestion.\n\n        Returns:\n            list(str). The list of html content strings.\n        '
        html_string_list = [self.change.new_value['html']]
        if self.change.old_value is not None:
            html_string_list.append(self.change.old_value['html'])
        return html_string_list

    def get_target_entity_html_strings(self) -> List[str]:
        if False:
            return 10
        'Gets all html content strings from target entity used in the\n        suggestion.\n\n        Returns:\n            list(str). The list of html content strings from target entity used\n            in the suggestion.\n        '
        if self.change.old_value is not None:
            return [self.change.old_value['html']]
        return []

    def convert_html_in_suggestion_change(self, conversion_fn: Callable[[str], str]) -> None:
        if False:
            return 10
        'Checks for HTML fields in a suggestion change and converts it\n        according to the conversion function.\n\n        Args:\n            conversion_fn: function. The function to be used for converting the\n                HTML.\n        '
        if self.change.old_value is not None:
            self.change.old_value['html'] = conversion_fn(self.change.old_value['html'])
        self.change.new_value['html'] = conversion_fn(self.change.new_value['html'])

class SuggestionTranslateContent(BaseSuggestion):
    """Domain object for a suggestion of type
    SUGGESTION_TYPE_TRANSLATE_CONTENT.
    """

    def __init__(self, suggestion_id: str, target_id: str, target_version_at_submission: int, status: str, author_id: str, final_reviewer_id: Optional[str], change: Mapping[str, change_domain.AcceptableChangeDictTypes], score_category: str, language_code: str, edited_by_reviewer: bool, last_updated: Optional[datetime.datetime]=None, created_on: Optional[datetime.datetime]=None) -> None:
        if False:
            return 10
        'Initializes an object of type SuggestionTranslateContent\n        corresponding to the SUGGESTION_TYPE_TRANSLATE_CONTENT choice.\n        '
        super().__init__(status, final_reviewer_id)
        self.suggestion_id = suggestion_id
        self.suggestion_type = feconf.SUGGESTION_TYPE_TRANSLATE_CONTENT
        self.target_type = feconf.ENTITY_TYPE_EXPLORATION
        self.target_id = target_id
        self.target_version_at_submission = target_version_at_submission
        self.author_id = author_id
        self.change: exp_domain.AddWrittenTranslationCmd = exp_domain.AddWrittenTranslationCmd(change)
        self.score_category = score_category
        self.language_code = language_code
        self.last_updated = last_updated
        self.created_on = created_on
        self.edited_by_reviewer = edited_by_reviewer
        self.image_context = feconf.IMAGE_CONTEXT_EXPLORATION_SUGGESTIONS

    def validate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Validates a suggestion object of type SuggestionTranslateContent.\n\n        Raises:\n            ValidationError. One or more attributes of the\n                SuggestionTranslateContent object are invalid.\n        '
        super().validate()
        if not isinstance(self.change, exp_domain.ExplorationChange):
            raise utils.ValidationError('Expected change to be an ExplorationChange, received %s' % type(self.change))
        if self.get_score_type() != suggestion_models.SCORE_TYPE_TRANSLATION:
            raise utils.ValidationError('Expected the first part of score_category to be %s , received %s' % (suggestion_models.SCORE_TYPE_TRANSLATION, self.get_score_type()))
        accepted_cmds = [exp_domain.DEPRECATED_CMD_ADD_TRANSLATION, exp_domain.CMD_ADD_WRITTEN_TRANSLATION]
        if self.change.cmd not in accepted_cmds:
            raise utils.ValidationError('Expected cmd to be %s, received %s' % (exp_domain.CMD_ADD_WRITTEN_TRANSLATION, self.change.cmd))
        if not utils.is_supported_audio_language_code(self.change.language_code):
            raise utils.ValidationError('Invalid language_code: %s' % self.change.language_code)
        if isinstance(self.change.translation_html, str):
            html_cleaner.validate_rte_tags(self.change.translation_html)
        if self.language_code is None:
            raise utils.ValidationError('language_code cannot be None')
        if self.language_code != self.change.language_code:
            raise utils.ValidationError('Expected language_code to be %s, received %s' % (self.change.language_code, self.language_code))

    def pre_update_validate(self, change: exp_domain.ExplorationChange) -> None:
        if False:
            return 10
        'Performs the pre update validation. This function needs to be called\n        before updating the suggestion.\n\n        Args:\n            change: ExplorationChange. The new change.\n\n        Raises:\n            ValidationError. Invalid new change.\n        '
        if self.change.cmd != change.cmd:
            raise utils.ValidationError('The new change cmd must be equal to %s' % self.change.cmd)
        if self.change.state_name != change.state_name:
            raise utils.ValidationError('The new change state_name must be equal to %s' % self.change.state_name)
        if self.change.content_html != change.content_html:
            raise utils.ValidationError('The new change content_html must be equal to %s' % self.change.content_html)
        if self.change.language_code != change.language_code:
            raise utils.ValidationError('The language code must be equal to %s' % self.change.language_code)

    def pre_accept_validate(self) -> None:
        if False:
            i = 10
            return i + 15
        'Performs referential validation. This function needs to be called\n        before accepting the suggestion.\n        '
        self.validate()
        exploration = exp_fetchers.get_exploration_by_id(self.target_id)
        if self.change.state_name not in exploration.states:
            raise utils.ValidationError('Expected %s to be a valid state name' % self.change.state_name)

    def accept(self, unused_commit_message: str) -> None:
        if False:
            return 10
        'Accepts the suggestion.'
        translated_content = translation_domain.TranslatedContent(self.change.translation_html, translation_domain.TranslatableContentFormat(self.change.data_format), needs_update=False)
        translation_services.add_new_translation(feconf.TranslatableEntityType.EXPLORATION, self.target_id, self.target_version_at_submission, self.language_code, self.change.content_id, translated_content)
        opportunity_services.update_translation_opportunity_with_accepted_suggestion(self.target_id, self.language_code)
        assert self.final_reviewer_id is not None
        if hasattr(self.change, 'data_format') and translation_domain.TranslatableContentFormat.is_data_format_list(self.change.data_format):
            return
        self._copy_new_images_to_target_entity_storage()

    def get_all_html_content_strings(self) -> List[str]:
        if False:
            while True:
                i = 10
        'Gets all html content strings used in this suggestion.\n\n        Returns:\n            list(str). The list of html content strings.\n        '
        content_strings = []
        if isinstance(self.change.translation_html, list):
            content_strings.extend(self.change.translation_html)
        else:
            content_strings.append(self.change.translation_html)
        if isinstance(self.change.content_html, list):
            content_strings.extend(self.change.content_html)
        else:
            content_strings.append(self.change.content_html)
        return content_strings

    def get_target_entity_html_strings(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        'Gets all html content strings from target entity used in the\n        suggestion.\n\n        Returns:\n            list(str). The list of html content strings from target entity used\n            in the suggestion.\n        '
        return [self.change.content_html]

    def convert_html_in_suggestion_change(self, conversion_fn: Callable[[str], str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Checks for HTML fields in a suggestion change and converts it\n        according to the conversion function.\n\n        Args:\n            conversion_fn: function. The function to be used for converting the\n                HTML.\n        '
        self.change.content_html = conversion_fn(self.change.content_html)
        self.change.translation_html = conversion_fn(self.change.translation_html)

class SuggestionAddQuestion(BaseSuggestion):
    """Domain object for a suggestion of type SUGGESTION_TYPE_ADD_QUESTION.

    Attributes:
        suggestion_id: str. The ID of the suggestion.
        suggestion_type: str. The type of the suggestion.
        target_type: str. The type of target entity being edited, for this
            subclass, target type is 'skill'.
        target_id: str. The ID of the skill the question was submitted to.
        target_version_at_submission: int. The version number of the target
            topic at the time of creation of the suggestion.
        status: str. The status of the suggestion.
        author_id: str. The ID of the user who submitted the suggestion.
        final_reviewer_id: str. The ID of the reviewer who has accepted/rejected
            the suggestion.
        change_cmd: QuestionChange. The change associated with the suggestion.
        score_category: str. The scoring category for the suggestion.
        last_updated: datetime.datetime. Date and time when the suggestion
            was last updated.
        language_code: str. The ISO 639-1 code used to query suggestions
            by language. In this case it is the language code of the question.
        edited_by_reviewer: bool. Whether the suggestion is edited by the
            reviewer.
    """

    def __init__(self, suggestion_id: str, target_id: str, target_version_at_submission: int, status: str, author_id: str, final_reviewer_id: Optional[str], change: Mapping[str, change_domain.AcceptableChangeDictTypes], score_category: str, language_code: str, edited_by_reviewer: bool, last_updated: Optional[datetime.datetime]=None, created_on: Optional[datetime.datetime]=None) -> None:
        if False:
            return 10
        'Initializes an object of type SuggestionAddQuestion\n        corresponding to the SUGGESTION_TYPE_ADD_QUESTION choice.\n        '
        super().__init__(status, final_reviewer_id)
        self.suggestion_id = suggestion_id
        self.suggestion_type = feconf.SUGGESTION_TYPE_ADD_QUESTION
        self.target_type = feconf.ENTITY_TYPE_SKILL
        self.target_id = target_id
        self.target_version_at_submission = target_version_at_submission
        self.author_id = author_id
        self.change: question_domain.CreateNewFullySpecifiedQuestionSuggestionCmd = question_domain.CreateNewFullySpecifiedQuestionSuggestionCmd(change)
        self.score_category = score_category
        self.language_code = language_code
        self.last_updated = last_updated
        self.created_on = created_on
        self.image_context = feconf.IMAGE_CONTEXT_QUESTION_SUGGESTIONS
        self._update_change_to_latest_state_schema_version()
        self.edited_by_reviewer = edited_by_reviewer

    def _update_change_to_latest_state_schema_version(self) -> None:
        if False:
            return 10
        'Holds the responsibility of performing a step-by-step, sequential\n        update of the state structure inside the change_cmd based on the schema\n        version of the current state dictionary.\n\n        Raises:\n            Exception. The state_schema_version of suggestion cannot be\n                processed.\n        '
        question_dict: question_domain.QuestionDict = self.change.question_dict
        state_schema_version = question_dict['question_state_data_schema_version']
        versioned_question_state: question_domain.VersionedQuestionStateDict = {'state_schema_version': state_schema_version, 'state': copy.deepcopy(question_dict['question_state_data'])}
        if not 25 <= state_schema_version <= feconf.CURRENT_STATE_SCHEMA_VERSION:
            raise utils.ValidationError('Expected state schema version to be in between 25 and %d, received %s.' % (feconf.CURRENT_STATE_SCHEMA_VERSION, state_schema_version))
        while state_schema_version < feconf.CURRENT_STATE_SCHEMA_VERSION:
            question_domain.Question.update_state_from_model(versioned_question_state, state_schema_version)
            state_schema_version += 1
        self.change.question_dict['question_state_data'] = versioned_question_state['state']
        self.change.question_dict['question_state_data_schema_version'] = state_schema_version

    def validate(self) -> None:
        if False:
            while True:
                i = 10
        'Validates a suggestion object of type SuggestionAddQuestion.\n\n        Raises:\n            ValidationError. One or more attributes of the SuggestionAddQuestion\n                object are invalid.\n        '
        super().validate()
        if self.get_score_type() != suggestion_models.SCORE_TYPE_QUESTION:
            raise utils.ValidationError('Expected the first part of score_category to be "%s" , received "%s"' % (suggestion_models.SCORE_TYPE_QUESTION, self.get_score_type()))
        if not isinstance(self.change, question_domain.QuestionSuggestionChange):
            raise utils.ValidationError('Expected change to be an instance of QuestionSuggestionChange')
        if not self.change.cmd:
            raise utils.ValidationError('Expected change to contain cmd')
        if self.change.cmd != question_domain.CMD_CREATE_NEW_FULLY_SPECIFIED_QUESTION:
            raise utils.ValidationError('Expected cmd to be %s, obtained %s' % (question_domain.CMD_CREATE_NEW_FULLY_SPECIFIED_QUESTION, self.change.cmd))
        if not self.change.question_dict:
            raise utils.ValidationError('Expected change to contain question_dict')
        question_dict: question_domain.QuestionDict = self.change.question_dict
        if self.language_code != constants.DEFAULT_LANGUAGE_CODE:
            raise utils.ValidationError('Expected language_code to be %s, received %s' % (constants.DEFAULT_LANGUAGE_CODE, self.language_code))
        if self.language_code != question_dict['language_code']:
            raise utils.ValidationError('Expected question language_code(%s) to be same as suggestion language_code(%s)' % (question_dict['language_code'], self.language_code))
        if not self.change.skill_difficulty:
            raise utils.ValidationError('Expected change to contain skill_difficulty')
        skill_difficulties = list(constants.SKILL_DIFFICULTY_LABEL_TO_FLOAT.values())
        if self._get_skill_difficulty() not in skill_difficulties:
            raise utils.ValidationError('Expected change skill_difficulty to be one of %s, found %s ' % (skill_difficulties, self._get_skill_difficulty()))
        question = question_domain.Question(None, state_domain.State.from_dict(self.change.question_dict['question_state_data']), self.change.question_dict['question_state_data_schema_version'], self.change.question_dict['language_code'], None, self.change.question_dict['linked_skill_ids'], self.change.question_dict['inapplicable_skill_misconception_ids'], self.change.question_dict['next_content_id_index'])
        question_state_data_schema_version = question_dict['question_state_data_schema_version']
        if question_state_data_schema_version != feconf.CURRENT_STATE_SCHEMA_VERSION:
            raise utils.ValidationError('Expected question state schema version to be %s, received %s' % (feconf.CURRENT_STATE_SCHEMA_VERSION, question_state_data_schema_version))
        question.partial_validate()

    def pre_accept_validate(self) -> None:
        if False:
            i = 10
            return i + 15
        'Performs referential validation. This function needs to be called\n        before accepting the suggestion.\n        '
        if self.change.skill_id is None:
            raise utils.ValidationError('Expected change to contain skill_id')
        self.validate()
        skill_domain.Skill.require_valid_skill_id(self.change.skill_id)
        skill = skill_fetchers.get_skill_by_id(self.change.skill_id, strict=False)
        if skill is None:
            raise utils.ValidationError("The skill with the given id doesn't exist.")

    def accept(self, unused_commit_message: str) -> None:
        if False:
            i = 10
            return i + 15
        'Accepts the suggestion.\n\n        Args:\n            unused_commit_message: str. This parameter is passed in for\n                consistency with the existing suggestions. As a default commit\n                message is used in the add_question function, the arg is unused.\n        '
        question_dict: question_domain.QuestionDict = self.change.question_dict
        question_dict['version'] = 1
        question_dict['id'] = question_services.get_new_question_id()
        question_dict['linked_skill_ids'] = [self.change.skill_id]
        question = question_domain.Question.from_dict(question_dict)
        question.validate()
        new_image_filenames = self.get_new_image_filenames_added_in_suggestion()
        if question.question_state_data.interaction.id == 'ImageClickInput':
            customization_arg_image_dict = cast(domain.ImageAndRegionDict, question.question_state_data.interaction.customization_args['imageAndRegions'].value)
            new_image_filenames.append(customization_arg_image_dict['imagePath'])
        fs_services.copy_images(self.image_context, self.target_id, feconf.ENTITY_TYPE_QUESTION, question_dict['id'], new_image_filenames)
        question_services.add_question(self.author_id, question)
        skill = skill_fetchers.get_skill_by_id(self.change.skill_id, strict=False)
        if skill is None:
            raise utils.ValidationError("The skill with the given id doesn't exist.")
        question_services.create_new_question_skill_link(self.author_id, question_dict['id'], self.change.skill_id, self._get_skill_difficulty())

    def populate_old_value_of_change(self) -> None:
        if False:
            while True:
                i = 10
        'Populates old value of the change.'
        pass

    def pre_update_validate(self, change: Union[question_domain.CreateNewFullySpecifiedQuestionSuggestionCmd, question_domain.CreateNewFullySpecifiedQuestionCmd]) -> None:
        if False:
            i = 10
            return i + 15
        'Performs the pre update validation. This functions need to be called\n        before updating the suggestion.\n\n        Args:\n            change: QuestionChange. The new change.\n\n        Raises:\n            ValidationError. Invalid new change.\n        '
        if self.change.cmd != change.cmd:
            raise utils.ValidationError('The new change cmd must be equal to %s' % self.change.cmd)
        if self.change.skill_id != change.skill_id:
            raise utils.ValidationError('The new change skill_id must be equal to %s' % self.change.skill_id)
        if self.change.skill_difficulty == change.skill_difficulty and self.change.question_dict == change.question_dict:
            raise utils.ValidationError('At least one of the new skill_difficulty or question_dict should be changed.')

    def _get_skill_difficulty(self) -> float:
        if False:
            return 10
        "Returns the suggestion's skill difficulty."
        return self.change.skill_difficulty

    def get_all_html_content_strings(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        'Gets all html content strings used in this suggestion.\n\n        Returns:\n            list(str). The list of html content strings.\n        '
        question_dict: question_domain.QuestionDict = self.change.question_dict
        state_object = state_domain.State.from_dict(question_dict['question_state_data'])
        html_string_list = state_object.get_all_html_content_strings()
        return html_string_list

    def get_target_entity_html_strings(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        'Gets all html content strings from target entity used in the\n        suggestion.\n        '
        return []

    def convert_html_in_suggestion_change(self, conversion_fn: Callable[[str], str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Checks for HTML fields in the suggestion change and converts it\n        according to the conversion function.\n\n        Args:\n            conversion_fn: function. The function to be used for converting the\n                HTML.\n        '
        question_dict: question_domain.QuestionDict = self.change.question_dict
        question_dict['question_state_data'] = state_domain.State.convert_html_fields_in_state(question_dict['question_state_data'], conversion_fn, state_uses_old_interaction_cust_args_schema=question_dict['question_state_data_schema_version'] < 38, state_uses_old_rule_template_schema=question_dict['question_state_data_schema_version'] < 45)
SUGGESTION_TYPES_TO_DOMAIN_CLASSES: Dict[str, Union[Type[SuggestionEditStateContent], Type[SuggestionTranslateContent], Type[SuggestionAddQuestion]]] = {feconf.SUGGESTION_TYPE_EDIT_STATE_CONTENT: SuggestionEditStateContent, feconf.SUGGESTION_TYPE_TRANSLATE_CONTENT: SuggestionTranslateContent, feconf.SUGGESTION_TYPE_ADD_QUESTION: SuggestionAddQuestion}

class CommunityContributionStats:
    """Domain object for the CommunityContributionStatsModel.

    Attributes:
        translation_reviewer_counts_by_lang_code: dict. A dictionary where the
            keys represent the language codes that translation suggestions are
            offered in and the values correspond to the total number of
            reviewers who have permission to review translation suggestions in
            that language.
        translation_suggestion_counts_by_lang_code: dict. A dictionary where
            the keys represent the language codes that translation suggestions
            are offered in and the values correspond to the total number of
            translation suggestions that are currently in review in that
            language.
        question_reviewer_count: int. The total number of reviewers who have
            permission to review question suggestions.
        question_suggestion_count: int. The total number of question
            suggestions that are currently in review.
    """

    def __init__(self, translation_reviewer_counts_by_lang_code: Dict[str, int], translation_suggestion_counts_by_lang_code: Dict[str, int], question_reviewer_count: int, question_suggestion_count: int) -> None:
        if False:
            print('Hello World!')
        self.translation_reviewer_counts_by_lang_code = translation_reviewer_counts_by_lang_code
        self.translation_suggestion_counts_by_lang_code = translation_suggestion_counts_by_lang_code
        self.question_reviewer_count = question_reviewer_count
        self.question_suggestion_count = question_suggestion_count

    def validate(self) -> None:
        if False:
            return 10
        'Validates the CommunityContributionStats object.\n\n        Raises:\n            ValidationError. One or more attributes of the\n                CommunityContributionStats object is invalid.\n        '
        for (language_code, reviewer_count) in self.translation_reviewer_counts_by_lang_code.items():
            if not utils.is_supported_audio_language_code(language_code):
                raise utils.ValidationError('Invalid language code for the translation reviewer counts: %s.' % language_code)
            if not isinstance(reviewer_count, int):
                raise utils.ValidationError('Expected the translation reviewer count to be an integer for %s language code, received: %s.' % (language_code, reviewer_count))
            if reviewer_count < 0:
                raise utils.ValidationError('Expected the translation reviewer count to be non-negative for %s language code, received: %s.' % (language_code, reviewer_count))
        for (language_code, suggestion_count) in self.translation_suggestion_counts_by_lang_code.items():
            if not utils.is_supported_audio_language_code(language_code):
                raise utils.ValidationError('Invalid language code for the translation suggestion counts: %s.' % language_code)
            if not isinstance(suggestion_count, int):
                raise utils.ValidationError('Expected the translation suggestion count to be an integer for %s language code, received: %s.' % (language_code, suggestion_count))
            if suggestion_count < 0:
                raise utils.ValidationError('Expected the translation suggestion count to be non-negative for %s language code, received: %s.' % (language_code, suggestion_count))
        if not isinstance(self.question_reviewer_count, int):
            raise utils.ValidationError('Expected the question reviewer count to be an integer, received: %s.' % self.question_reviewer_count)
        if self.question_reviewer_count < 0:
            raise utils.ValidationError('Expected the question reviewer count to be non-negative, received: %s.' % self.question_reviewer_count)
        if not isinstance(self.question_suggestion_count, int):
            raise utils.ValidationError('Expected the question suggestion count to be an integer, received: %s.' % self.question_suggestion_count)
        if self.question_suggestion_count < 0:
            raise utils.ValidationError('Expected the question suggestion count to be non-negative, received: %s.' % self.question_suggestion_count)

    def set_translation_reviewer_count_for_language_code(self, language_code: str, count: int) -> None:
        if False:
            i = 10
            return i + 15
        'Sets the translation reviewer count to be count, for the given\n        language code.\n\n        Args:\n            language_code: str. The translation suggestion language code that\n                reviewers have the rights to review.\n            count: int. The number of reviewers that have the rights to review\n                translation suggestions in language_code.\n        '
        self.translation_reviewer_counts_by_lang_code[language_code] = count

    def set_translation_suggestion_count_for_language_code(self, language_code: str, count: int) -> None:
        if False:
            print('Hello World!')
        'Sets the translation suggestion count to be count, for the language\n        code given.\n\n        Args:\n            language_code: str. The translation suggestion language code.\n            count: int. The number of translation suggestions in language_code\n                that are currently in review.\n        '
        self.translation_suggestion_counts_by_lang_code[language_code] = count

    def are_translation_reviewers_needed_for_lang_code(self, lang_code: str) -> bool:
        if False:
            return 10
        'Returns whether or not more reviewers are needed to review\n        translation suggestions in the given language code. Translation\n        suggestions in a given language need more reviewers if the number of\n        translation suggestions in that language divided by the number of\n        translation reviewers in that language is greater than\n        ParamNames.MAX_NUMBER_OF_SUGGESTIONS_PER_REVIEWER.\n\n        Args:\n            lang_code: str. The language code of the translation\n                suggestions.\n\n        Returns:\n            bool. Whether or not more reviewers are needed to review\n            translation suggestions in the given language code.\n       '
        if lang_code not in self.translation_suggestion_counts_by_lang_code:
            return False
        if lang_code not in self.translation_reviewer_counts_by_lang_code:
            return True
        number_of_reviewers = self.translation_reviewer_counts_by_lang_code[lang_code]
        number_of_suggestions = self.translation_suggestion_counts_by_lang_code[lang_code]
        max_number_of_suggestions_per_reviewer = platform_feature_services.get_platform_parameter_value(platform_parameter_list.ParamNames.MAX_NUMBER_OF_SUGGESTIONS_PER_REVIEWER.value)
        assert isinstance(max_number_of_suggestions_per_reviewer, int)
        return bool(number_of_suggestions > max_number_of_suggestions_per_reviewer * number_of_reviewers)

    def get_translation_language_codes_that_need_reviewers(self) -> Set[str]:
        if False:
            while True:
                i = 10
        'Returns the language codes where more reviewers are needed to review\n        translations in those language codes. Translation suggestions in a\n        given language need more reviewers if the number of translation\n        suggestions in that language divided by the number of translation\n        reviewers in that language is greater than\n        ParamNames.MAX_NUMBER_OF_SUGGESTIONS_PER_REVIEWER.\n\n        Returns:\n            set. A set of of the language codes where more translation reviewers\n            are needed.\n        '
        language_codes_that_need_reviewers = set()
        for language_code in self.translation_suggestion_counts_by_lang_code:
            if self.are_translation_reviewers_needed_for_lang_code(language_code):
                language_codes_that_need_reviewers.add(language_code)
        return language_codes_that_need_reviewers

    def are_question_reviewers_needed(self) -> bool:
        if False:
            while True:
                i = 10
        'Returns whether or not more reviewers are needed to review question\n        suggestions. Question suggestions need more reviewers if the number of\n        question suggestions divided by the number of question reviewers is\n        greater than ParamNames.MAX_NUMBER_OF_SUGGESTIONS_PER_REVIEWER.\n\n        Returns:\n            bool. Whether or not more reviewers are needed to review\n            question suggestions.\n       '
        if self.question_suggestion_count == 0:
            return False
        if self.question_reviewer_count == 0:
            return True
        max_number_of_suggestions_per_reviewer = platform_feature_services.get_platform_parameter_value(platform_parameter_list.ParamNames.MAX_NUMBER_OF_SUGGESTIONS_PER_REVIEWER.value)
        assert isinstance(max_number_of_suggestions_per_reviewer, int)
        return bool(self.question_suggestion_count > max_number_of_suggestions_per_reviewer * self.question_reviewer_count)

class TranslationContributionStatsDict(TypedDict):
    """Dictionary representing the TranslationContributionStats object."""
    language_code: str
    contributor_user_id: str
    topic_id: str
    submitted_translations_count: int
    submitted_translation_word_count: int
    accepted_translations_count: int
    accepted_translations_without_reviewer_edits_count: int
    accepted_translation_word_count: int
    rejected_translations_count: int
    rejected_translation_word_count: int
    contribution_dates: Set[datetime.date]

class TranslationContributionStatsFrontendDict(TypedDict):
    """Dictionary representing the TranslationContributionStats
    object for frontend.
    """
    language_code: str
    topic_id: str
    submitted_translations_count: int
    submitted_translation_word_count: int
    accepted_translations_count: int
    accepted_translations_without_reviewer_edits_count: int
    accepted_translation_word_count: int
    rejected_translations_count: int
    rejected_translation_word_count: int
    first_contribution_date: str
    last_contribution_date: str

class TranslationContributionStats:
    """Domain object for the TranslationContributionStatsModel."""

    def __init__(self, language_code: str, contributor_user_id: str, topic_id: str, submitted_translations_count: int, submitted_translation_word_count: int, accepted_translations_count: int, accepted_translations_without_reviewer_edits_count: int, accepted_translation_word_count: int, rejected_translations_count: int, rejected_translation_word_count: int, contribution_dates: Set[datetime.date]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.language_code = language_code
        self.contributor_user_id = contributor_user_id
        self.topic_id = topic_id
        self.submitted_translations_count = submitted_translations_count
        self.submitted_translation_word_count = submitted_translation_word_count
        self.accepted_translations_count = accepted_translations_count
        self.accepted_translations_without_reviewer_edits_count = accepted_translations_without_reviewer_edits_count
        self.accepted_translation_word_count = accepted_translation_word_count
        self.rejected_translations_count = rejected_translations_count
        self.rejected_translation_word_count = rejected_translation_word_count
        self.contribution_dates = contribution_dates

    def to_dict(self) -> TranslationContributionStatsDict:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict representation of a TranslationContributionStats\n        domain object.\n\n        Returns:\n            dict. A dict representation of a TranslationContributionStats\n            domain object.\n        '
        return {'language_code': self.language_code, 'contributor_user_id': self.contributor_user_id, 'topic_id': self.topic_id, 'submitted_translations_count': self.submitted_translations_count, 'submitted_translation_word_count': self.submitted_translation_word_count, 'accepted_translations_count': self.accepted_translations_count, 'accepted_translations_without_reviewer_edits_count': self.accepted_translations_without_reviewer_edits_count, 'accepted_translation_word_count': self.accepted_translation_word_count, 'rejected_translations_count': self.rejected_translations_count, 'rejected_translation_word_count': self.rejected_translation_word_count, 'contribution_dates': self.contribution_dates}

    def to_frontend_dict(self) -> TranslationContributionStatsFrontendDict:
        if False:
            return 10
        'Returns a dict representation of a TranslationContributionStats\n        domain object for frontend.\n\n        Returns:\n            dict. A dict representation of a TranslationContributionStats\n            domain object for frontend.\n        '
        sorted_contribution_dates = sorted(self.contribution_dates)
        return {'language_code': self.language_code, 'topic_id': self.topic_id, 'submitted_translations_count': self.submitted_translations_count, 'submitted_translation_word_count': self.submitted_translation_word_count, 'accepted_translations_count': self.accepted_translations_count, 'accepted_translations_without_reviewer_edits_count': self.accepted_translations_without_reviewer_edits_count, 'accepted_translation_word_count': self.accepted_translation_word_count, 'rejected_translations_count': self.rejected_translations_count, 'rejected_translation_word_count': self.rejected_translation_word_count, 'first_contribution_date': sorted_contribution_dates[0].strftime('%b %Y'), 'last_contribution_date': sorted_contribution_dates[-1].strftime('%b %Y')}

class TranslationReviewStatsDict(TypedDict):
    """Dictionary representing the TranslationReviewStats object."""
    language_code: str
    contributor_user_id: str
    topic_id: str
    reviewed_translations_count: int
    reviewed_translation_word_count: int
    accepted_translations_count: int
    accepted_translation_word_count: int
    accepted_translations_with_reviewer_edits_count: int
    first_contribution_date: datetime.date
    last_contribution_date: datetime.date

class TranslationReviewStatsFrontendDict(TypedDict):
    """Dictionary representing the TranslationReviewStats
    object for frontend.
    """
    language_code: str
    topic_id: str
    reviewed_translations_count: int
    reviewed_translation_word_count: int
    accepted_translations_count: int
    accepted_translation_word_count: int
    accepted_translations_with_reviewer_edits_count: int
    first_contribution_date: str
    last_contribution_date: str

class TranslationReviewStats:
    """Domain object for the TranslationReviewStatsModel."""

    def __init__(self, language_code: str, contributor_user_id: str, topic_id: str, reviewed_translations_count: int, reviewed_translation_word_count: int, accepted_translations_count: int, accepted_translation_word_count: int, accepted_translations_with_reviewer_edits_count: int, first_contribution_date: datetime.date, last_contribution_date: datetime.date) -> None:
        if False:
            print('Hello World!')
        self.language_code = language_code
        self.contributor_user_id = contributor_user_id
        self.topic_id = topic_id
        self.reviewed_translations_count = reviewed_translations_count
        self.reviewed_translation_word_count = reviewed_translation_word_count
        self.accepted_translations_count = accepted_translations_count
        self.accepted_translation_word_count = accepted_translation_word_count
        self.accepted_translations_with_reviewer_edits_count = accepted_translations_with_reviewer_edits_count
        self.first_contribution_date = first_contribution_date
        self.last_contribution_date = last_contribution_date

    def to_dict(self) -> TranslationReviewStatsDict:
        if False:
            return 10
        'Returns a dict representation of a TranslationReviewStats\n        domain object.\n\n        Returns:\n            dict. A dict representation of a TranslationReviewStats\n            domain object.\n        '
        return {'language_code': self.language_code, 'contributor_user_id': self.contributor_user_id, 'topic_id': self.topic_id, 'reviewed_translations_count': self.reviewed_translations_count, 'reviewed_translation_word_count': self.reviewed_translation_word_count, 'accepted_translations_count': self.accepted_translations_count, 'accepted_translation_word_count': self.accepted_translation_word_count, 'accepted_translations_with_reviewer_edits_count': self.accepted_translations_with_reviewer_edits_count, 'first_contribution_date': self.first_contribution_date, 'last_contribution_date': self.last_contribution_date}

    def to_frontend_dict(self) -> TranslationReviewStatsFrontendDict:
        if False:
            while True:
                i = 10
        'Returns a dict representation of a TranslationReviewStats\n        domain object for frontend.\n\n        Returns:\n            dict. A dict representation of a TranslationReviewStats\n            domain object for frontend.\n        '
        return {'language_code': self.language_code, 'topic_id': self.topic_id, 'reviewed_translations_count': self.reviewed_translations_count, 'reviewed_translation_word_count': self.reviewed_translation_word_count, 'accepted_translations_count': self.accepted_translations_count, 'accepted_translation_word_count': self.accepted_translation_word_count, 'accepted_translations_with_reviewer_edits_count': self.accepted_translations_with_reviewer_edits_count, 'first_contribution_date': self.first_contribution_date.strftime('%b %Y'), 'last_contribution_date': self.last_contribution_date.strftime('%b %Y')}

class QuestionContributionStatsDict(TypedDict):
    """Dictionary representing the QuestionContributionStats object."""
    contributor_user_id: str
    topic_id: str
    submitted_questions_count: int
    accepted_questions_count: int
    accepted_questions_without_reviewer_edits_count: int
    first_contribution_date: datetime.date
    last_contribution_date: datetime.date

class QuestionContributionStatsFrontendDict(TypedDict):
    """Dictionary representing the QuestionContributionStats
    object for frontend.
    """
    topic_id: str
    submitted_questions_count: int
    accepted_questions_count: int
    accepted_questions_without_reviewer_edits_count: int
    first_contribution_date: str
    last_contribution_date: str

class QuestionContributionStats:
    """Domain object for the QuestionContributionStatsModel."""

    def __init__(self, contributor_user_id: str, topic_id: str, submitted_questions_count: int, accepted_questions_count: int, accepted_questions_without_reviewer_edits_count: int, first_contribution_date: datetime.date, last_contribution_date: datetime.date) -> None:
        if False:
            while True:
                i = 10
        self.contributor_user_id = contributor_user_id
        self.topic_id = topic_id
        self.submitted_questions_count = submitted_questions_count
        self.accepted_questions_count = accepted_questions_count
        self.accepted_questions_without_reviewer_edits_count = accepted_questions_without_reviewer_edits_count
        self.first_contribution_date = first_contribution_date
        self.last_contribution_date = last_contribution_date

    def to_dict(self) -> QuestionContributionStatsDict:
        if False:
            return 10
        'Returns a dict representation of a QuestionContributionStats\n        domain object.\n\n        Returns:\n            dict. A dict representation of a QuestionContributionStats\n            domain object.\n        '
        return {'contributor_user_id': self.contributor_user_id, 'topic_id': self.topic_id, 'submitted_questions_count': self.submitted_questions_count, 'accepted_questions_count': self.accepted_questions_count, 'accepted_questions_without_reviewer_edits_count': self.accepted_questions_without_reviewer_edits_count, 'first_contribution_date': self.first_contribution_date, 'last_contribution_date': self.last_contribution_date}

    def to_frontend_dict(self) -> QuestionContributionStatsFrontendDict:
        if False:
            i = 10
            return i + 15
        'Returns a dict representation of a QuestionContributionStats\n        domain object for frontend.\n\n        Returns:\n            dict. A dict representation of a QuestionContributionStats\n            domain object for frontend.\n        '
        return {'topic_id': self.topic_id, 'submitted_questions_count': self.submitted_questions_count, 'accepted_questions_count': self.accepted_questions_count, 'accepted_questions_without_reviewer_edits_count': self.accepted_questions_without_reviewer_edits_count, 'first_contribution_date': self.first_contribution_date.strftime('%b %Y'), 'last_contribution_date': self.last_contribution_date.strftime('%b %Y')}

class QuestionReviewStatsDict(TypedDict):
    """Dictionary representing the QuestionReviewStats object."""
    contributor_user_id: str
    topic_id: str
    reviewed_questions_count: int
    accepted_questions_count: int
    accepted_questions_with_reviewer_edits_count: int
    first_contribution_date: datetime.date
    last_contribution_date: datetime.date

class QuestionReviewStatsFrontendDict(TypedDict):
    """Dictionary representing the QuestionReviewStats
    object for frontend.
    """
    topic_id: str
    reviewed_questions_count: int
    accepted_questions_count: int
    accepted_questions_with_reviewer_edits_count: int
    first_contribution_date: str
    last_contribution_date: str

class QuestionReviewStats:
    """Domain object for the QuestionReviewStatsModel."""

    def __init__(self, contributor_user_id: str, topic_id: str, reviewed_questions_count: int, accepted_questions_count: int, accepted_questions_with_reviewer_edits_count: int, first_contribution_date: datetime.date, last_contribution_date: datetime.date) -> None:
        if False:
            i = 10
            return i + 15
        self.contributor_user_id = contributor_user_id
        self.topic_id = topic_id
        self.reviewed_questions_count = reviewed_questions_count
        self.accepted_questions_count = accepted_questions_count
        self.accepted_questions_with_reviewer_edits_count = accepted_questions_with_reviewer_edits_count
        self.first_contribution_date = first_contribution_date
        self.last_contribution_date = last_contribution_date

    def to_dict(self) -> QuestionReviewStatsDict:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict representation of a QuestionContributionStats\n        domain object.\n\n        Returns:\n            dict. A dict representation of a QuestionContributionStats\n            domain object.\n        '
        return {'contributor_user_id': self.contributor_user_id, 'topic_id': self.topic_id, 'reviewed_questions_count': self.reviewed_questions_count, 'accepted_questions_count': self.accepted_questions_count, 'accepted_questions_with_reviewer_edits_count': self.accepted_questions_with_reviewer_edits_count, 'first_contribution_date': self.first_contribution_date, 'last_contribution_date': self.last_contribution_date}

    def to_frontend_dict(self) -> QuestionReviewStatsFrontendDict:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict representation of a QuestionContributionStats\n        domain object for frontend.\n\n        Returns:\n            dict. A dict representation of a QuestionContributionStats\n            domain object for frontend.\n        '
        return {'topic_id': self.topic_id, 'reviewed_questions_count': self.reviewed_questions_count, 'accepted_questions_count': self.accepted_questions_count, 'accepted_questions_with_reviewer_edits_count': self.accepted_questions_with_reviewer_edits_count, 'first_contribution_date': self.first_contribution_date.strftime('%b %Y'), 'last_contribution_date': self.last_contribution_date.strftime('%b %Y')}

class ContributorCertificateInfoDict(TypedDict):
    """Dictionary representing the ContributorCertificateInfo object."""
    from_date: str
    to_date: str
    team_lead: str
    contribution_hours: str
    language: Optional[str]

class ContributorCertificateInfo:
    """Encapsulates key information that is used to generate contributor
    certificate.
    """

    def __init__(self, from_date: str, to_date: str, team_lead: str, contribution_hours: str, language: Optional[str]) -> None:
        if False:
            return 10
        self.from_date = from_date
        self.to_date = to_date
        self.team_lead = team_lead
        self.contribution_hours = contribution_hours
        self.language = language

    def to_dict(self) -> ContributorCertificateInfoDict:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict representation of a ContributorCertificateInfo\n        domain object.\n\n        Returns:\n            dict. A dict representation of a ContributorCertificateInfo\n            domain object.\n        '
        return {'from_date': self.from_date, 'to_date': self.to_date, 'team_lead': self.team_lead, 'contribution_hours': self.contribution_hours, 'language': self.language}

class ContributorMilestoneEmailInfo:
    """Encapsulates key information that is used to create the email content for
    notifying contributors about milestones they achieved.

    Attributes:
        contributor_user_id: str. The ID of the contributor.
        language_code: str|None. The language code of the suggestion.
        contribution_type: str. The type of the contribution i.e.
            translation or question.
        contribution_sub_type: str. The sub type of the contribution
            i.e. submissions/acceptances/reviews/edits.
        rank_name: str. The name of the rank that the contributor achieved.
    """

    def __init__(self, contributor_user_id: str, contribution_type: str, contribution_subtype: str, language_code: Optional[str], rank_name: str) -> None:
        if False:
            i = 10
            return i + 15
        self.contributor_user_id = contributor_user_id
        self.contribution_type = contribution_type
        self.contribution_subtype = contribution_subtype
        self.language_code = language_code
        self.rank_name = rank_name

class ContributorStatsSummaryDict(TypedDict):
    """Dictionary representing the ContributorStatsSummary object."""
    contributor_user_id: str
    translation_contribution_stats: List[TranslationContributionStatsDict]
    question_contribution_stats: List[QuestionContributionStatsDict]
    translation_review_stats: List[TranslationReviewStatsDict]
    question_review_stats: List[QuestionReviewStatsDict]

class ContributorStatsSummary:
    """Encapsulates key information that is used to send to the frontend
    regarding contributor stats.

    Attributes:
        contributor_user_id: str. The ID of the contributor.
        translation_contribution_stats: list(TranslationContributionStats). A
            list of TranslationContributionStats corresponding to the user.
        question_contribution_stats: list(QuestionContributionStats). A list of
            QuestionContributionStats corresponding to the user.
        translation_review_stats: list(TranslationReviewStats). A list of
            TranslationReviewStats corresponding to the user.
        question_review_stats: list(QuestionReviewStats). A list of
            QuestionReviewStats  corresponding to the user.
    """

    def __init__(self, contributor_user_id: str, translation_contribution_stats: List[TranslationContributionStats], question_contribution_stats: List[QuestionContributionStats], translation_review_stats: List[TranslationReviewStats], question_review_stats: List[QuestionReviewStats]) -> None:
        if False:
            print('Hello World!')
        self.contributor_user_id = contributor_user_id
        self.translation_contribution_stats = translation_contribution_stats
        self.question_contribution_stats = question_contribution_stats
        self.translation_review_stats = translation_review_stats
        self.question_review_stats = question_review_stats

    def to_dict(self) -> ContributorStatsSummaryDict:
        if False:
            while True:
                i = 10
        'Returns a dict representation of a ContributorStatsSummary\n        domain object.\n\n        Returns:\n            dict. A dict representation of a ContributorStatsSummary\n            domain object.\n        '
        return {'contributor_user_id': self.contributor_user_id, 'translation_contribution_stats': [stats.to_dict() for stats in self.translation_contribution_stats], 'question_contribution_stats': [stats.to_dict() for stats in self.question_contribution_stats], 'translation_review_stats': [stats.to_dict() for stats in self.translation_review_stats], 'question_review_stats': [stats.to_dict() for stats in self.question_review_stats]}

class ReviewableSuggestionEmailInfo:
    """Encapsulates key information that is used to create the email content for
    notifying admins and reviewers that there are suggestions that need to be
    reviewed.

    Attributes:
        suggestion_type: str. The type of the suggestion.
        language_code: str. The language code of the suggestion.
        suggestion_content: str. The suggestion content that is emphasized for
            a user when they are viewing a list of suggestions on the
            Contributor Dashboard.
        submission_datetime: datetime.datetime. Date and time when the
            suggestion was submitted for review.
    """

    def __init__(self, suggestion_type: str, language_code: str, suggestion_content: str, submission_datetime: datetime.datetime) -> None:
        if False:
            print('Hello World!')
        self.suggestion_type = suggestion_type
        self.language_code = language_code
        self.suggestion_content = suggestion_content
        self.submission_datetime = submission_datetime

class TranslationSubmitterTotalContributionStatsFrontendDict(TypedDict):
    """Dictionary representing the TranslationSubmitterTotalContributionStats
    object for frontend.
    """
    language_code: str
    contributor_name: str
    topic_names: List[str]
    recent_performance: int
    overall_accuracy: float
    submitted_translations_count: int
    submitted_translation_word_count: int
    accepted_translations_count: int
    accepted_translations_without_reviewer_edits_count: int
    accepted_translation_word_count: int
    rejected_translations_count: int
    rejected_translation_word_count: int
    first_contribution_date: str
    last_contributed_in_days: int

class TranslationSubmitterTotalContributionStats:
    """Domain object for the TranslationSubmitterTotalContributionStatsModel."""

    def __init__(self, language_code: str, contributor_id: str, topic_ids_with_translation_submissions: List[str], recent_review_outcomes: List[str], recent_performance: int, overall_accuracy: float, submitted_translations_count: int, submitted_translation_word_count: int, accepted_translations_count: int, accepted_translations_without_reviewer_edits_count: int, accepted_translation_word_count: int, rejected_translations_count: int, rejected_translation_word_count: int, first_contribution_date: datetime.date, last_contribution_date: datetime.date) -> None:
        if False:
            while True:
                i = 10
        self.language_code = language_code
        self.contributor_id = contributor_id
        self.topic_ids_with_translation_submissions = topic_ids_with_translation_submissions
        self.recent_review_outcomes = recent_review_outcomes
        self.recent_performance = recent_performance
        self.overall_accuracy = overall_accuracy
        self.submitted_translations_count = submitted_translations_count
        self.submitted_translation_word_count = submitted_translation_word_count
        self.accepted_translations_count = accepted_translations_count
        self.accepted_translations_without_reviewer_edits_count = accepted_translations_without_reviewer_edits_count
        self.accepted_translation_word_count = accepted_translation_word_count
        self.rejected_translations_count = rejected_translations_count
        self.rejected_translation_word_count = rejected_translation_word_count
        self.first_contribution_date = first_contribution_date
        self.last_contribution_date = last_contribution_date

    def to_frontend_dict(self) -> TranslationSubmitterTotalContributionStatsFrontendDict:
        if False:
            return 10
        'Returns a dict representation of a\n        TranslationSubmitterTotalContributionStats domain object for frontend.\n\n        Returns:\n            dict. The dict representation.\n        '
        topic_summaries = topic_fetchers.get_multi_topic_summaries(self.topic_ids_with_translation_submissions)
        topic_name_by_topic_id = []
        for topic_summary in topic_summaries:
            if topic_summary is not None:
                topic_name_by_topic_id.append(topic_summary.name)
        contributor_name = user_services.get_username(self.contributor_id)
        return {'language_code': self.language_code, 'contributor_name': contributor_name, 'topic_names': topic_name_by_topic_id, 'recent_performance': self.recent_performance, 'overall_accuracy': self.overall_accuracy, 'submitted_translations_count': self.submitted_translations_count, 'submitted_translation_word_count': self.submitted_translation_word_count, 'accepted_translations_count': self.accepted_translations_count, 'accepted_translations_without_reviewer_edits_count': self.accepted_translations_without_reviewer_edits_count, 'accepted_translation_word_count': self.accepted_translation_word_count, 'rejected_translations_count': self.rejected_translations_count, 'rejected_translation_word_count': self.rejected_translation_word_count, 'first_contribution_date': self.first_contribution_date.strftime('%b %d, %Y'), 'last_contributed_in_days': int((datetime.date.today() - self.last_contribution_date).days)}

class TranslationReviewerTotalContributionStatsFrontendDict(TypedDict):
    """Dictionary representing the TranslationReviewerTotalContributionStats
    object for frontend.
    """
    language_code: str
    contributor_name: str
    topic_names: List[str]
    reviewed_translations_count: int
    accepted_translations_count: int
    accepted_translations_with_reviewer_edits_count: int
    accepted_translation_word_count: int
    rejected_translations_count: int
    first_contribution_date: str
    last_contributed_in_days: int

class TranslationReviewerTotalContributionStats:
    """Domain object for the TranslationReviewerTotalContributionStats."""

    def __init__(self, language_code: str, contributor_id: str, topic_ids_with_translation_reviews: List[str], reviewed_translations_count: int, accepted_translations_count: int, accepted_translations_with_reviewer_edits_count: int, accepted_translation_word_count: int, rejected_translations_count: int, first_contribution_date: datetime.date, last_contribution_date: datetime.date) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.language_code = language_code
        self.contributor_id = contributor_id
        self.topic_ids_with_translation_reviews = topic_ids_with_translation_reviews
        self.reviewed_translations_count = reviewed_translations_count
        self.accepted_translations_count = accepted_translations_count
        self.accepted_translations_with_reviewer_edits_count = accepted_translations_with_reviewer_edits_count
        self.accepted_translation_word_count = accepted_translation_word_count
        self.rejected_translations_count = rejected_translations_count
        self.first_contribution_date = first_contribution_date
        self.last_contribution_date = last_contribution_date

    def to_frontend_dict(self) -> TranslationReviewerTotalContributionStatsFrontendDict:
        if False:
            while True:
                i = 10
        'Returns a dict representation of a\n        TranslationReviewerTotalContributionStats domain object for frontend.\n\n        Returns:\n            dict. The dict representation.\n        '
        topic_summaries = topic_fetchers.get_multi_topic_summaries(self.topic_ids_with_translation_reviews)
        topic_name_by_topic_id = []
        for topic_summary in topic_summaries:
            if topic_summary is not None:
                topic_name_by_topic_id.append(topic_summary.name)
        contributor_name = user_services.get_username(self.contributor_id)
        return {'language_code': self.language_code, 'contributor_name': contributor_name, 'topic_names': topic_name_by_topic_id, 'reviewed_translations_count': self.reviewed_translations_count, 'accepted_translations_count': self.accepted_translations_count, 'accepted_translations_with_reviewer_edits_count': self.accepted_translations_with_reviewer_edits_count, 'accepted_translation_word_count': self.accepted_translation_word_count, 'rejected_translations_count': self.rejected_translations_count, 'first_contribution_date': self.first_contribution_date.strftime('%b %d, %Y'), 'last_contributed_in_days': int((datetime.date.today() - self.last_contribution_date).days)}

class QuestionSubmitterTotalContributionStatsFrontendDict(TypedDict):
    """Dictionary representing the QuestionSubmitterTotalContributionStats
    object for frontend.
    """
    contributor_name: str
    topic_names: List[str]
    recent_performance: int
    overall_accuracy: float
    submitted_questions_count: int
    accepted_questions_count: int
    accepted_questions_without_reviewer_edits_count: int
    rejected_questions_count: int
    first_contribution_date: str
    last_contributed_in_days: int

class QuestionSubmitterTotalContributionStats:
    """Domain object for the QuestionSubmitterTotalContributionStats."""

    def __init__(self, contributor_id: str, topic_ids_with_question_submissions: List[str], recent_review_outcomes: List[str], recent_performance: int, overall_accuracy: float, submitted_questions_count: int, accepted_questions_count: int, accepted_questions_without_reviewer_edits_count: int, rejected_questions_count: int, first_contribution_date: datetime.date, last_contribution_date: datetime.date) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.contributor_id = contributor_id
        self.topic_ids_with_question_submissions = topic_ids_with_question_submissions
        self.recent_review_outcomes = recent_review_outcomes
        self.recent_performance = recent_performance
        self.overall_accuracy = overall_accuracy
        self.submitted_questions_count = submitted_questions_count
        self.accepted_questions_count = accepted_questions_count
        self.accepted_questions_without_reviewer_edits_count = accepted_questions_without_reviewer_edits_count
        self.rejected_questions_count = rejected_questions_count
        self.first_contribution_date = first_contribution_date
        self.last_contribution_date = last_contribution_date

    def to_frontend_dict(self) -> QuestionSubmitterTotalContributionStatsFrontendDict:
        if False:
            i = 10
            return i + 15
        'Returns a dict representation of a\n        QuestionSubmitterTotalContributionStats domain object for frontend.\n\n        Returns:\n            dict. The dict representation.\n        '
        topic_summaries = topic_fetchers.get_multi_topic_summaries(self.topic_ids_with_question_submissions)
        topic_name_by_topic_id = []
        for topic_summary in topic_summaries:
            if topic_summary is not None:
                topic_name_by_topic_id.append(topic_summary.name)
        contributor_name = user_services.get_username(self.contributor_id)
        return {'contributor_name': contributor_name, 'topic_names': topic_name_by_topic_id, 'recent_performance': self.recent_performance, 'overall_accuracy': self.overall_accuracy, 'submitted_questions_count': self.submitted_questions_count, 'accepted_questions_count': self.accepted_questions_count, 'accepted_questions_without_reviewer_edits_count': self.accepted_questions_without_reviewer_edits_count, 'rejected_questions_count': self.rejected_questions_count, 'first_contribution_date': self.first_contribution_date.strftime('%b %d, %Y'), 'last_contributed_in_days': int((datetime.date.today() - self.last_contribution_date).days)}

class QuestionReviewerTotalContributionStatsFrontendDict(TypedDict):
    """Dictionary representing the QuestionReviewerTotalContributionStats
    object for frontend.
    """
    contributor_name: str
    topic_names: List[str]
    reviewed_questions_count: int
    accepted_questions_count: int
    accepted_questions_with_reviewer_edits_count: int
    rejected_questions_count: int
    first_contribution_date: str
    last_contributed_in_days: int

class QuestionReviewerTotalContributionStats:
    """Domain object for the QuestionReviewerTotalContributionStats."""

    def __init__(self, contributor_id: str, topic_ids_with_question_reviews: List[str], reviewed_questions_count: int, accepted_questions_count: int, accepted_questions_with_reviewer_edits_count: int, rejected_questions_count: int, first_contribution_date: datetime.date, last_contribution_date: datetime.date) -> None:
        if False:
            return 10
        self.contributor_id = contributor_id
        self.topic_ids_with_question_reviews = topic_ids_with_question_reviews
        self.reviewed_questions_count = reviewed_questions_count
        self.accepted_questions_count = accepted_questions_count
        self.accepted_questions_with_reviewer_edits_count = accepted_questions_with_reviewer_edits_count
        self.rejected_questions_count = rejected_questions_count
        self.first_contribution_date = first_contribution_date
        self.last_contribution_date = last_contribution_date

    def to_frontend_dict(self) -> QuestionReviewerTotalContributionStatsFrontendDict:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict representation of a\n        questionReviewerTotalContributionStats domain object for frontend.\n\n        Returns:\n            dict. The dict representation.\n        '
        topic_summaries = topic_fetchers.get_multi_topic_summaries(self.topic_ids_with_question_reviews)
        topic_name_by_topic_id = []
        for topic_summary in topic_summaries:
            if topic_summary is not None:
                topic_name_by_topic_id.append(topic_summary.name)
        contributor_name = user_services.get_username(self.contributor_id)
        return {'contributor_name': contributor_name, 'topic_names': topic_name_by_topic_id, 'reviewed_questions_count': self.reviewed_questions_count, 'accepted_questions_count': self.accepted_questions_count, 'accepted_questions_with_reviewer_edits_count': self.accepted_questions_with_reviewer_edits_count, 'rejected_questions_count': self.rejected_questions_count, 'first_contribution_date': self.first_contribution_date.strftime('%b %d, %Y'), 'last_contributed_in_days': int((datetime.date.today() - self.last_contribution_date).days)}