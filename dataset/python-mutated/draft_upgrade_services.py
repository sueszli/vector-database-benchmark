"""Commands that can be used to upgrade draft to newer Exploration versions."""
from __future__ import annotations
import logging
from core import utils
from core.domain import exp_domain
from core.domain import html_validation_service
from core.domain import rules_registry
from core.domain import state_domain
from core.platform import models
from typing import Callable, List, Optional, Union, cast
MYPY = False
if MYPY:
    from mypy_imports import exp_models
(exp_models,) = models.Registry.import_models([models.Names.EXPLORATION])
AllowedDraftChangeListTypes = Union[state_domain.SubtitledHtmlDict, state_domain.CustomizationArgsDictType, state_domain.OutcomeDict, List[state_domain.HintDict], state_domain.SolutionDict, List[state_domain.AnswerGroupDict], str]

class InvalidDraftConversionException(Exception):
    """Error class for invalid draft conversion. Should be raised in a draft
    conversion function if it is not possible to upgrade a draft, and indicates
    that try_upgrading_draft_to_exp_version should return None.
    """
    pass

def try_upgrading_draft_to_exp_version(draft_change_list: List[exp_domain.ExplorationChange], current_draft_version: int, to_exp_version: int, exp_id: str) -> Optional[List[exp_domain.ExplorationChange]]:
    if False:
        return 10
    'Try upgrading a list of ExplorationChange domain objects to match the\n    latest exploration version.\n\n    For now, this handles the scenario where all commits between\n    current_draft_version and to_exp_version migrate only the state schema.\n\n    Args:\n        draft_change_list: list(ExplorationChange). The list of\n            ExplorationChange domain objects to upgrade.\n        current_draft_version: int. Current draft version.\n        to_exp_version: int. Target exploration version.\n        exp_id: str. Exploration id.\n\n    Returns:\n        list(ExplorationChange) or None. A list of ExplorationChange domain\n        objects after upgrade or None if upgrade fails.\n\n    Raises:\n        InvalidInputException. The current_draft_version is greater than\n            to_exp_version.\n    '
    if current_draft_version > to_exp_version:
        raise utils.InvalidInputException('Current draft version is greater than the exploration version.')
    if current_draft_version == to_exp_version:
        return None
    exp_versions = list(range(current_draft_version + 1, to_exp_version + 1))
    commits_list = exp_models.ExplorationCommitLogEntryModel.get_multi(exp_id, exp_versions)
    upgrade_times = 0
    while current_draft_version + upgrade_times < to_exp_version:
        commit = commits_list[upgrade_times]
        assert commit is not None
        if len(commit.commit_cmds) != 1 or commit.commit_cmds[0]['cmd'] != exp_domain.CMD_MIGRATE_STATES_SCHEMA_TO_LATEST_VERSION:
            return None
        conversion_fn_name = '_convert_states_v%s_dict_to_v%s_dict' % (commit.commit_cmds[0]['from_version'], commit.commit_cmds[0]['to_version'])
        if not hasattr(DraftUpgradeUtil, conversion_fn_name):
            logging.warning('%s is not implemented' % conversion_fn_name)
            return None
        conversion_fn = getattr(DraftUpgradeUtil, conversion_fn_name)
        try:
            draft_change_list = conversion_fn(draft_change_list)
        except InvalidDraftConversionException:
            return None
        upgrade_times += 1
    return draft_change_list

class DraftUpgradeUtil:
    """Wrapper class that contains util functions to upgrade drafts."""

    @classmethod
    def _convert_html_in_draft_change_list(cls, draft_change_list: List[exp_domain.ExplorationChange], conversion_fn: Callable[[str], str]) -> List[exp_domain.ExplorationChange]:
        if False:
            print('Hello World!')
        'Applies a conversion function on all HTML fields in the provided\n        draft change list.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n            conversion_fn: function. The function to be used for converting the\n                HTML.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n        '
        for (i, change) in enumerate(draft_change_list):
            if not change.cmd == exp_domain.CMD_EDIT_STATE_PROPERTY:
                continue
            new_value: AllowedDraftChangeListTypes = change.new_value
            if change.property_name == exp_domain.STATE_PROPERTY_CONTENT:
                edit_content_property_cmd = cast(exp_domain.EditExpStatePropertyContentCmd, change)
                new_value = edit_content_property_cmd.new_value
                new_value['html'] = conversion_fn(new_value['html'])
            elif change.property_name == exp_domain.STATE_PROPERTY_INTERACTION_CUST_ARGS:
                edit_interaction_cust_args_cmd = cast(exp_domain.EditExpStatePropertyInteractionCustArgsCmd, change)
                new_value = edit_interaction_cust_args_cmd.new_value
                if 'choices' in new_value.keys():
                    subtitled_html_new_value_dicts = cast(List[state_domain.SubtitledHtmlDict], new_value['choices']['value'])
                    for (value_index, value) in enumerate(subtitled_html_new_value_dicts):
                        if isinstance(value, dict) and 'html' in value:
                            subtitled_html_new_value_dicts[value_index]['html'] = conversion_fn(value['html'])
                        elif isinstance(value, str):
                            subtitled_html_new_value_dicts[value_index] = conversion_fn(value)
            elif change.property_name == 'written_translations':
                translations_mapping = change.new_value['translations_mapping']
                assert isinstance(translations_mapping, dict)
                for (content_id, language_code_to_written_translation) in translations_mapping.items():
                    for language_code in language_code_to_written_translation.keys():
                        translation_dict = translations_mapping[content_id][language_code]
                        if 'html' in translation_dict:
                            translations_mapping[content_id][language_code]['html'] = conversion_fn(translations_mapping[content_id][language_code]['html'])
            elif change.property_name == exp_domain.STATE_PROPERTY_INTERACTION_DEFAULT_OUTCOME and new_value is not None:
                edit_interaction_default_outcome_cmd = cast(exp_domain.EditExpStatePropertyInteractionDefaultOutcomeCmd, change)
                new_value = state_domain.Outcome.convert_html_in_outcome(edit_interaction_default_outcome_cmd.new_value, conversion_fn)
            elif change.property_name == exp_domain.STATE_PROPERTY_INTERACTION_HINTS:
                edit_interaction_hints_cmd = cast(exp_domain.EditExpStatePropertyInteractionHintsCmd, change)
                hint_dicts = edit_interaction_hints_cmd.new_value
                new_value = [state_domain.Hint.convert_html_in_hint(hint_dict, conversion_fn) for hint_dict in hint_dicts]
            elif change.property_name == exp_domain.STATE_PROPERTY_INTERACTION_SOLUTION and new_value is not None:
                edit_interaction_solution_cmd = cast(exp_domain.EditExpStatePropertyInteractionSolutionCmd, change)
                new_value = edit_interaction_solution_cmd.new_value
                new_value['explanation']['html'] = conversion_fn(new_value['explanation']['html'])
                if new_value['correct_answer']:
                    if isinstance(new_value['correct_answer'], list):
                        for (list_index, html_list) in enumerate(new_value['correct_answer']):
                            if isinstance(html_list, list):
                                for (answer_html_index, answer_html) in enumerate(html_list):
                                    if isinstance(answer_html, str):
                                        correct_answer = cast(List[List[str]], new_value['correct_answer'])
                                        correct_answer[list_index][answer_html_index] = conversion_fn(answer_html)
            elif change.property_name == exp_domain.STATE_PROPERTY_INTERACTION_ANSWER_GROUPS:
                html_field_types_to_rule_specs = rules_registry.Registry.get_html_field_types_to_rule_specs(state_schema_version=41)
                edit_interaction_answer_groups_cmd = cast(exp_domain.EditExpStatePropertyInteractionAnswerGroupsCmd, change)
                answer_group_dicts = edit_interaction_answer_groups_cmd.new_value
                new_value = [state_domain.AnswerGroup.convert_html_in_answer_group(answer_group, conversion_fn, html_field_types_to_rule_specs) for answer_group in answer_group_dicts]
            if new_value is not None:
                draft_change_list[i] = exp_domain.ExplorationChange({'cmd': change.cmd, 'property_name': change.property_name, 'state_name': change.state_name, 'new_value': new_value})
        return draft_change_list

    @classmethod
    def _convert_states_v54_dict_to_v55_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            i = 10
            return i + 15
        "Converts draft change list from state version 54 to 55. Version 55\n        changes content ids for content and removes written_translation property\n        form the state, converting draft to anew version won't be possible.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n\n        Raises:\n            InvalidDraftConversionException. The conversion cannot be\n                completed.\n        "
        for exp_change in draft_change_list:
            if exp_change.cmd == exp_domain.CMD_EDIT_STATE_PROPERTY:
                raise InvalidDraftConversionException('Conversion cannot be completed.')
        return draft_change_list

    @classmethod
    def _convert_states_v53_dict_to_v54_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            while True:
                i = 10
        "Converts draft change list from state version 53 to 54. State\n        version 54 adds catchMisspellings customization_arg to TextInput\n        interaction. As this is a new property and therefore\n        doesn't affect any pre-existing drafts, there should be\n        no changes to drafts.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n        "
        return draft_change_list

    @classmethod
    def _convert_states_v52_dict_to_v53_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            while True:
                i = 10
        "Converts from version 52 to 53. Version 53 fixes general\n        state, interaction and rte data. This will update the drafts\n        for state and RTE part but won't be able to do for interaction.\n        The `ExplorationChange` object for interaction is divided into\n        further properties and we won't be able to collect enough info\n        to update the draft.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n\n        Raises:\n            InvalidDraftConversionException. The conversion cannot be\n                completed.\n        "
        drafts_to_remove = [exp_domain.STATE_PROPERTY_INTERACTION_CUST_ARGS, exp_domain.STATE_PROPERTY_INTERACTION_STICKY, exp_domain.STATE_PROPERTY_INTERACTION_HANDLERS, exp_domain.STATE_PROPERTY_INTERACTION_ANSWER_GROUPS, exp_domain.STATE_PROPERTY_INTERACTION_DEFAULT_OUTCOME, exp_domain.STATE_PROPERTY_INTERACTION_HINTS, exp_domain.STATE_PROPERTY_INTERACTION_SOLUTION]
        for exp_change in draft_change_list:
            if exp_change.cmd != exp_domain.CMD_EDIT_STATE_PROPERTY:
                continue
            if exp_change.property_name in drafts_to_remove:
                raise InvalidDraftConversionException('Conversion cannot be completed.')
            if exp_change.property_name == exp_domain.STATE_PROPERTY_CONTENT:
                assert isinstance(exp_change.new_value, dict)
                html = exp_change.new_value['html']
                html = exp_domain.Exploration.fix_content(html)
                exp_change.new_value['html'] = html
            elif exp_change.property_name == exp_domain.DEPRECATED_STATE_PROPERTY_WRITTEN_TRANSLATIONS:
                assert isinstance(exp_change.new_value, dict)
                written_translations = exp_change.new_value
                for translations in written_translations['translations_mapping'].values():
                    for written_translation in translations.values():
                        if written_translation['data_format'] == 'html':
                            if isinstance(written_translation['translation'], list):
                                raise InvalidDraftConversionException('Conversion cannot be completed.')
                            fixed_translation = exp_domain.Exploration.fix_content(written_translation['translation'])
                            written_translation['translation'] = fixed_translation
        return draft_change_list

    @classmethod
    def _convert_states_v51_dict_to_v52_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            i = 10
            return i + 15
        "Converts from version 51 to 52. Version 52 fixes content IDs\n        in translations and voiceovers (some content IDs are removed).\n        We discard drafts that work with content IDs to make sure that they\n        don't contain content IDs that were removed.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n\n        Raises:\n            InvalidDraftConversionException. The conversion cannot be\n                completed.\n        "
        for exp_change in draft_change_list:
            if exp_change.cmd in (exp_domain.DEPRECATED_CMD_MARK_WRITTEN_TRANSLATIONS_AS_NEEDING_UPDATE, exp_domain.DEPRECATED_CMD_MARK_WRITTEN_TRANSLATION_AS_NEEDING_UPDATE, exp_domain.CMD_ADD_WRITTEN_TRANSLATION, exp_domain.DEPRECATED_CMD_ADD_TRANSLATION):
                raise InvalidDraftConversionException('Conversion cannot be completed.')
            if exp_change.cmd == exp_domain.CMD_EDIT_STATE_PROPERTY:
                if exp_change.property_name == exp_domain.DEPRECATED_STATE_PROPERTY_NEXT_CONTENT_ID_INDEX:
                    raise InvalidDraftConversionException('Conversion cannot be completed.')
        return draft_change_list

    @classmethod
    def _convert_states_v50_dict_to_v51_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            while True:
                i = 10
        'Converts draft change list from state version 50 to 51. Adds\n        a new field dest_if_really_stuck to Outcome class to direct the\n        learner to a custom revision state.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n        '
        for (i, change) in enumerate(draft_change_list):
            if change.cmd == exp_domain.CMD_EDIT_STATE_PROPERTY and change.property_name == exp_domain.STATE_PROPERTY_INTERACTION_ANSWER_GROUPS:
                edit_interaction_answer_groups_cmd = cast(exp_domain.EditExpStatePropertyInteractionAnswerGroupsCmd, change)
                new_answer_groups_dicts = edit_interaction_answer_groups_cmd.new_value
                answer_group_dicts: List[state_domain.AnswerGroupDict] = []
                for answer_group_dict in new_answer_groups_dicts:
                    outcome_dict: state_domain.OutcomeDict = {'dest': answer_group_dict['outcome']['dest'], 'dest_if_really_stuck': None, 'feedback': answer_group_dict['outcome']['feedback'], 'labelled_as_correct': answer_group_dict['outcome']['labelled_as_correct'], 'param_changes': answer_group_dict['outcome']['param_changes'], 'refresher_exploration_id': answer_group_dict['outcome']['refresher_exploration_id'], 'missing_prerequisite_skill_id': answer_group_dict['outcome']['missing_prerequisite_skill_id']}
                    answer_group_dicts.append({'rule_specs': answer_group_dict['rule_specs'], 'outcome': outcome_dict, 'training_data': answer_group_dict['training_data'], 'tagged_skill_misconception_id': None})
                draft_change_list[i] = exp_domain.ExplorationChange({'cmd': exp_domain.CMD_EDIT_STATE_PROPERTY, 'property_name': exp_domain.STATE_PROPERTY_INTERACTION_ANSWER_GROUPS, 'state_name': change.state_name, 'new_value': answer_group_dicts})
            elif change.property_name == exp_domain.STATE_PROPERTY_INTERACTION_DEFAULT_OUTCOME and change.cmd == exp_domain.CMD_EDIT_STATE_PROPERTY:
                edit_interaction_default_outcome_cmd = cast(exp_domain.EditExpStatePropertyInteractionDefaultOutcomeCmd, change)
                new_default_outcome_dict = edit_interaction_default_outcome_cmd.new_value
                default_outcome_dict: state_domain.OutcomeDict = {'dest': new_default_outcome_dict['dest'], 'dest_if_really_stuck': None, 'feedback': new_default_outcome_dict['feedback'], 'labelled_as_correct': new_default_outcome_dict['labelled_as_correct'], 'param_changes': new_default_outcome_dict['param_changes'], 'refresher_exploration_id': new_default_outcome_dict['refresher_exploration_id'], 'missing_prerequisite_skill_id': new_default_outcome_dict['missing_prerequisite_skill_id']}
                draft_change_list[i] = exp_domain.ExplorationChange({'cmd': exp_domain.CMD_EDIT_STATE_PROPERTY, 'property_name': exp_domain.STATE_PROPERTY_INTERACTION_DEFAULT_OUTCOME, 'state_name': change.state_name, 'new_value': default_outcome_dict})
        return draft_change_list

    @classmethod
    def _convert_states_v49_dict_to_v50_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            i = 10
            return i + 15
        'Converts draft change list from state version 49 to 50. State\n        version 50 removes rules from explorations that use one of the following\n        rules: [ContainsSomeOf, OmitsSomeOf, MatchesWithGeneralForm]. It also\n        renames `customOskLetters` cust arg to `allowedVariables`. This should\n        not affect drafts.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n        '
        return draft_change_list

    @classmethod
    def _convert_states_v48_dict_to_v49_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            return 10
        'Converts draft change list from state version 48 to 49. State\n        version 49 adds requireNonnegativeInput customization_arg to\n        NumericInput interaction.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n        '
        return draft_change_list

    @classmethod
    def _convert_states_v47_dict_to_v48_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            i = 10
            return i + 15
        'Converts draft change list from state version 47 to 48. State\n        version 48 fixes encoding issues in HTML fields.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n        '
        conversion_fn = html_validation_service.fix_incorrectly_encoded_chars
        return cls._convert_html_in_draft_change_list(draft_change_list, conversion_fn)

    @classmethod
    def _convert_states_v46_dict_to_v47_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            for i in range(10):
                print('nop')
        'Converts draft change list from state version 46 to 47. State\n        version 47 deprecates oppia-noninteractive-svgdiagram tag and converts\n        existing occurences of it to oppia-noninteractive-image tag.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n        '
        conversion_fn = html_validation_service.convert_svg_diagram_tags_to_image_tags
        return cls._convert_html_in_draft_change_list(draft_change_list, conversion_fn)

    @classmethod
    def _convert_states_v45_dict_to_v46_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            while True:
                i = 10
        "Converts draft change list from state version 45 to 46. State\n        version 46 ensures that written translations corresponding to\n        unicode text have data_format field set to 'unicode' and that they\n        do not contain any HTML tags. This should not affect drafts.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n        "
        return draft_change_list

    @classmethod
    def _convert_states_v44_dict_to_v45_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            return 10
        "Converts draft change list from state version 44 to 45. State\n        version 45 adds a linked skill id property to the\n        state. As this is a new property and therefore doesn't affect any\n        pre-existing drafts, there should be no changes to drafts.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n        "
        return draft_change_list

    @classmethod
    def _convert_states_v43_dict_to_v44_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            while True:
                i = 10
        'Converts draft change list from state version 43 to 44. State\n        version 44 adds card_is_checkpoint boolean variable to the\n        state, for which there should be no changes to drafts.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n        '
        return draft_change_list

    @classmethod
    def _convert_states_v42_dict_to_v43_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            for i in range(10):
                print('nop')
        'Converts draft change list from state version 42 to 43.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n\n        Raises:\n            InvalidDraftConversionException. The conversion cannot be\n                completed.\n        '
        for change in draft_change_list:
            if change.property_name == exp_domain.STATE_PROPERTY_INTERACTION_ANSWER_GROUPS:
                raise InvalidDraftConversionException('Conversion cannot be completed.')
        return draft_change_list

    @classmethod
    def _convert_states_v41_dict_to_v42_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            return 10
        'Converts draft change list from state version 41 to 42.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n\n        Raises:\n            InvalidDraftConversionException. The conversion cannot be\n                completed.\n        '
        for change in draft_change_list:
            if change.property_name == exp_domain.STATE_PROPERTY_INTERACTION_ANSWER_GROUPS:
                raise InvalidDraftConversionException('Conversion cannot be completed.')
        return draft_change_list

    @classmethod
    def _convert_states_v40_dict_to_v41_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            for i in range(10):
                print('nop')
        'Converts draft change list from state version 40 to 41.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n\n        Raises:\n            InvalidDraftConversionException. The conversion cannot be\n                completed.\n        '
        for change in draft_change_list:
            if change.property_name == exp_domain.STATE_PROPERTY_INTERACTION_ANSWER_GROUPS:
                raise InvalidDraftConversionException('Conversion cannot be completed.')
        return draft_change_list

    @classmethod
    def _convert_states_v39_dict_to_v40_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            return 10
        'Converts draft change list from state version 39 to 40.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n\n        Raises:\n            InvalidDraftConversionException. The conversion cannot be\n                completed.\n        '
        for change in draft_change_list:
            if change.property_name == exp_domain.STATE_PROPERTY_INTERACTION_CUST_ARGS:
                raise InvalidDraftConversionException('Conversion cannot be completed.')
        return draft_change_list

    @classmethod
    def _convert_states_v38_dict_to_v39_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            print('Hello World!')
        'Converts draft change list from state version 38 to 39. State\n        version 39 adds a customization arg for the Numeric Expression Input\n        interactions that allows creators to modify the placeholder text,\n        for which there should be no changes to drafts.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n        '
        return draft_change_list

    @classmethod
    def _convert_states_v37_dict_to_v38_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            while True:
                i = 10
        'Converts draft change list from state version 37 to 38. State\n        version 38 adds a customization arg for the Math interactions that\n        allows creators to specify the letters that would be displayed to the\n        learner, for which there should be no changes to drafts.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n        '
        return draft_change_list

    @classmethod
    def _convert_states_v36_dict_to_v37_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            i = 10
            return i + 15
        'Converts draft change list from version 36 to 37.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n        '
        for change in draft_change_list:
            if change.property_name == exp_domain.STATE_PROPERTY_INTERACTION_ANSWER_GROUPS:
                edit_interaction_answer_groups_cmd = cast(exp_domain.EditExpStatePropertyInteractionAnswerGroupsCmd, change)
                answer_group_dicts = edit_interaction_answer_groups_cmd.new_value
                for answer_group_dict in answer_group_dicts:
                    for rule_spec_dict in answer_group_dict['rule_specs']:
                        if rule_spec_dict['rule_type'] == 'CaseSensitiveEquals':
                            rule_spec_dict['rule_type'] = 'Equals'
        return draft_change_list

    @classmethod
    def _convert_states_v35_dict_to_v36_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            print('Hello World!')
        'Converts draft change list from version 35 to 36.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n\n        Raises:\n            InvalidDraftConversionException. Conversion cannot be completed.\n        '
        for change in draft_change_list:
            if change.property_name == exp_domain.STATE_PROPERTY_INTERACTION_CUST_ARGS:
                raise InvalidDraftConversionException('Conversion cannot be completed.')
        return draft_change_list

    @classmethod
    def _convert_states_v34_dict_to_v35_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            while True:
                i = 10
        'Converts draft change list from state version 34 to 35. State\n        version 35 upgrades all explorations that use the MathExpressionInput\n        interaction to use one of AlgebraicExpressionInput,\n        NumericExpressionInput, or MathEquationInput interactions. There should\n        be no changes to the draft for this migration.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n\n        Raises:\n            InvalidDraftConversionException. Conversion cannot be completed.\n        '
        for change in draft_change_list:
            interaction_id_change_condition = change.property_name == exp_domain.STATE_PROPERTY_INTERACTION_ID and change.new_value == 'MathExpressionInput'
            answer_groups_change_condition = change.property_name == exp_domain.STATE_PROPERTY_INTERACTION_ANSWER_GROUPS and isinstance(change.new_value, list) and (change.new_value[0]['rule_specs'][0]['rule_type'] == 'IsMathematicallyEquivalentTo')
            if interaction_id_change_condition or answer_groups_change_condition:
                raise InvalidDraftConversionException('Conversion cannot be completed.')
        return draft_change_list

    @classmethod
    def _convert_states_v33_dict_to_v34_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            for i in range(10):
                print('nop')
        'Converts draft change list from state version 33 to 34. State\n        version 34 adds the new schema for Math components.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n        '
        conversion_fn = html_validation_service.add_math_content_to_math_rte_components
        return cls._convert_html_in_draft_change_list(draft_change_list, conversion_fn)

    @classmethod
    def _convert_states_v32_dict_to_v33_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            i = 10
            return i + 15
        'Converts draft change list from state version 32 to 33. State\n        version 33 adds showChoicesInShuffledOrder boolean variable to the\n        MultipleChoiceInput interaction.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n        '
        for (i, change) in enumerate(draft_change_list):
            if change.cmd == exp_domain.CMD_EDIT_STATE_PROPERTY and change.property_name == exp_domain.STATE_PROPERTY_INTERACTION_CUST_ARGS:
                assert isinstance(change.new_value, dict)
                if list(change.new_value.keys()) == ['choices']:
                    change.new_value['showChoicesInShuffledOrder'] = {'value': False}
                    draft_change_list[i] = exp_domain.ExplorationChange({'cmd': exp_domain.CMD_EDIT_STATE_PROPERTY, 'property_name': exp_domain.STATE_PROPERTY_INTERACTION_CUST_ARGS, 'state_name': change.state_name, 'new_value': change.new_value})
        return draft_change_list

    @classmethod
    def _convert_states_v31_dict_to_v32_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            while True:
                i = 10
        'Converts draft change list from state version 31 to 32. State\n        version 32 adds a customization arg for the "Add" button text\n        in SetInput interaction, for which there should be no changes\n        to drafts.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n        '
        return draft_change_list

    @classmethod
    def _convert_states_v30_dict_to_v31_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            while True:
                i = 10
        'Converts draft change list from state version 30 to 31. State\n        Version 31 adds the duration_secs float for the Voiceover\n        section of state.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n        '
        for (i, change) in enumerate(draft_change_list):
            if change.cmd == exp_domain.CMD_EDIT_STATE_PROPERTY and change.property_name == exp_domain.STATE_PROPERTY_RECORDED_VOICEOVERS:
                edit_recorded_voiceovers_cmd = cast(exp_domain.EditExpStatePropertyRecordedVoiceoversCmd, change)
                recorded_voiceovers_dict = edit_recorded_voiceovers_cmd.new_value
                new_voiceovers_mapping = recorded_voiceovers_dict['voiceovers_mapping']
                language_codes_to_audio_metadata = new_voiceovers_mapping.values()
                for language_codes in language_codes_to_audio_metadata:
                    for audio_metadata in language_codes.values():
                        audio_metadata['duration_secs'] = 0.0
                draft_change_list[i] = exp_domain.ExplorationChange({'cmd': exp_domain.CMD_EDIT_STATE_PROPERTY, 'property_name': exp_domain.STATE_PROPERTY_RECORDED_VOICEOVERS, 'state_name': change.state_name, 'new_value': {'voiceovers_mapping': new_voiceovers_mapping}})
        return draft_change_list

    @classmethod
    def _convert_states_v29_dict_to_v30_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            print('Hello World!')
        'Converts draft change list from state version 29 to 30. State\n        version 30 replaces tagged_misconception_id with\n        tagged_skill_misconception_id.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n        '
        for (i, change) in enumerate(draft_change_list):
            if change.cmd == exp_domain.CMD_EDIT_STATE_PROPERTY and change.property_name == exp_domain.STATE_PROPERTY_INTERACTION_ANSWER_GROUPS:
                edit_interaction_answer_groups_cmd = cast(exp_domain.EditExpStatePropertyInteractionAnswerGroupsCmd, change)
                new_answer_groups_dicts = edit_interaction_answer_groups_cmd.new_value
                answer_group_dicts: List[state_domain.AnswerGroupDict] = []
                for answer_group_dict in new_answer_groups_dicts:
                    answer_group_dicts.append({'rule_specs': answer_group_dict['rule_specs'], 'outcome': answer_group_dict['outcome'], 'training_data': answer_group_dict['training_data'], 'tagged_skill_misconception_id': None})
                draft_change_list[i] = exp_domain.ExplorationChange({'cmd': exp_domain.CMD_EDIT_STATE_PROPERTY, 'property_name': exp_domain.STATE_PROPERTY_INTERACTION_ANSWER_GROUPS, 'state_name': change.state_name, 'new_value': answer_group_dicts})
        return draft_change_list

    @classmethod
    def _convert_states_v28_dict_to_v29_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            for i in range(10):
                print('nop')
        'Converts draft change list from state version 28 to 29. State\n        version 29 adds solicit_answer_details boolean variable to the\n        state, for which there should be no changes to drafts.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n        '
        return draft_change_list

    @classmethod
    def _convert_states_v27_dict_to_v28_dict(cls, draft_change_list: List[exp_domain.ExplorationChange]) -> List[exp_domain.ExplorationChange]:
        if False:
            print('Hello World!')
        'Converts draft change list from state version 27 to 28. State\n        version 28 replaces content_ids_to_audio_translations with\n        recorded_voiceovers.\n\n        Args:\n            draft_change_list: list(ExplorationChange). The list of\n                ExplorationChange domain objects to upgrade.\n\n        Returns:\n            list(ExplorationChange). The converted draft_change_list.\n        '
        for (i, change) in enumerate(draft_change_list):
            if change.cmd == exp_domain.CMD_EDIT_STATE_PROPERTY and change.property_name == exp_domain.STATE_PROPERTY_CONTENT_IDS_TO_AUDIO_TRANSLATIONS_DEPRECATED:
                content_ids_to_audio_translations_cmd = cast(exp_domain.EditExpStatePropertyContentIdsToAudioTranslationsDeprecatedCmd, change)
                voiceovers_dict = content_ids_to_audio_translations_cmd.new_value
                draft_change_list[i] = exp_domain.ExplorationChange({'cmd': exp_domain.CMD_EDIT_STATE_PROPERTY, 'property_name': exp_domain.STATE_PROPERTY_RECORDED_VOICEOVERS, 'state_name': change.state_name, 'new_value': {'voiceovers_mapping': voiceovers_dict}})
        return draft_change_list