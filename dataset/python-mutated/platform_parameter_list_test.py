"""Tests for registered platform parameters."""
from __future__ import annotations
from core.domain import platform_feature_services as feature_services
from core.domain import platform_parameter_registry
from core.tests import test_utils

class ExistingPlatformParameterValidityTests(test_utils.GenericTestBase):
    """Tests to validate platform parameters registered in
    core/domain/platform_parameter_list.py.
    """
    EXPECTED_PARAM_NAMES = ['always_ask_learners_for_answer_details', 'android_beta_landing_page', 'blog_pages', 'cd_admin_dashboard_new_ui', 'checkpoint_celebration', 'contributor_dashboard_accomplishments', 'contributor_dashboard_reviewer_emails_is_enabled', 'diagnostic_test', 'dummy_feature_flag_for_e2e_tests', 'dummy_parameter', 'email_footer', 'email_sender_name', 'enable_admin_notifications_for_reviewer_shortage', 'end_chapter_celebration', 'high_bounce_rate_task_minimum_exploration_starts', 'high_bounce_rate_task_state_bounce_rate_creation_threshold', 'high_bounce_rate_task_state_bounce_rate_obsoletion_threshold', 'is_improvements_tab_enabled', 'learner_groups_are_enabled', 'max_number_of_suggestions_per_reviewer', 'max_number_of_tags_assigned_to_blog_post', 'notify_admins_suggestions_waiting_too_long_is_enabled', 'promo_bar_enabled', 'promo_bar_message', 'record_playthrough_probability', 'serial_chapter_launch_curriculum_admin_view', 'serial_chapter_launch_learner_view', 'show_feedback_updates_in_profile_pic_dropdown', 'show_redesigned_learner_dashboard', 'show_translation_size', 'signup_email_body_content', 'signup_email_subject_content', 'unpublish_exploration_email_html_body']

    def test_all_defined_parameters_are_valid(self) -> None:
        if False:
            i = 10
            return i + 15
        all_names = platform_parameter_registry.Registry.get_all_platform_parameter_names()
        for name in all_names:
            param = platform_parameter_registry.Registry.get_platform_parameter(name)
            param.validate()

    def test_number_of_parameters_meets_expectation(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that the Registry and EXPECTED_PARAM_NAMES have the same number\n        of platform parameters.\n\n        If this test fails, it means either:\n            - There are parameters defined in\n                core/domain/platform_parameter_list.py but not added to\n                EXPECTED_PARAM_NAMES above.\n            - There are parameters accidentally deleted from\n                core/domain/platform_parameter_list.py.\n        If you are defining new platform parameters, make sure to add it to the\n        EXPECTED_PARAM_NAMES list as well.\n        '
        self.assertEqual(len(platform_parameter_registry.Registry.get_all_platform_parameter_names()), len(self.EXPECTED_PARAM_NAMES))

    def test_all_expected_parameters_are_present_in_registry(self) -> None:
        if False:
            return 10
        "Test that all parameters in EXPECTED_PARAM_NAMES are present in\n        Registry.\n\n        If this test fails, it means some parameters in EXPECTED_PARAM_NAMES\n        are missing in the registry. It's most likely caused by accidentally\n        deleting some parameters in core/domain/platform_parameter_list.py.\n\n        To fix this, please make sure no parameter is deleted. If you really\n        need to delete a parameter (this should not happen in most cases),\n        make sure it's also deleted from EXPECTED_PARAM_NAMES.\n        "
        existing_names = platform_parameter_registry.Registry.get_all_platform_parameter_names()
        missing_names = set(self.EXPECTED_PARAM_NAMES) - set(existing_names)
        self.assertFalse(missing_names, msg='Platform parameters missing in registry: %s.' % list(missing_names))

    def test_no_unexpected_parameter_in_registry(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test that all parameters registered in Registry are expected.\n\n        If this test fails, it means some parameters in\n        core/domain/platform_parameter_list.py are not found in\n        EXPECTED_PARAM_NAMES.\n\n        If you are creating new platform parameters, make sure to add it to\n        the EXPECTED_PARAM_NAMES list as well.\n        '
        existing_names = platform_parameter_registry.Registry.get_all_platform_parameter_names()
        unexpected_names = set(existing_names) - set(self.EXPECTED_PARAM_NAMES)
        self.assertFalse(unexpected_names, msg='Unexpected platform parameters: %s.' % list(unexpected_names))

    def test_all_feature_flags_are_of_bool_type(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        feature_flags = feature_services.get_all_feature_flag_dicts()
        self.assertGreater(len(feature_flags), 0)
        for feature in feature_flags:
            self.assertEqual(feature['data_type'], 'bool', 'We expect all the feature-flags to be of type boolean but "%s" feature-flag is of type "%s".' % (feature['name'], feature['data_type']))

    def test_all_feature_flags_have_default_value_as_false(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        feature_flags = feature_services.get_all_feature_flag_dicts()
        self.assertGreater(len(feature_flags), 0)
        for feature in feature_flags:
            self.assertEqual(feature['default_value'], False, 'We expect all the feature-flags default_value to be False but "%s" feature-flag has "%s".' % (feature['name'], feature['default_value']))