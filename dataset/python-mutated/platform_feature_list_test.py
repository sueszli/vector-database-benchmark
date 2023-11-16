"""Test of platform feature list."""
from __future__ import annotations
import os
import re
from core import platform_feature_list
from core import utils
from core.domain import platform_parameter_domain
from core.domain import platform_parameter_registry as registry
from core.tests import test_utils
from typing import Final, List
FRONTEND_FEATURE_NAMES_PATH: Final = os.path.join(os.getcwd(), 'core/templates/domain/platform_feature', 'feature-status-summary.model.ts')
ENUM_BODY_REGEXP: Final = re.compile('enum FeatureNames \\{(.+?)\\}', flags=re.DOTALL)
ENUM_MEMBER_REGEXP: Final = re.compile("([a-zA-Z0-9_]+?)\\s+=\\s+\\'([a-zA-Z0-9_]+?)\\'")

class PlatformFeatureListTest(test_utils.GenericTestBase):
    """Tests for feature flags listed in platform_feature_list.py."""

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        super().setUp()
        self.all_features_list = platform_feature_list.DEV_FEATURES_LIST + platform_feature_list.TEST_FEATURES_LIST + platform_feature_list.PROD_FEATURES_LIST
        self.all_features_set = set(self.all_features_list)

    def _parse_feature_names_in_frontend(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        'Reads and parses feature flag definition in frontend.'
        with utils.open_file(FRONTEND_FEATURE_NAMES_PATH, 'r') as f:
            content = f.read()
        body_content = ENUM_BODY_REGEXP.search(content)
        assert body_content is not None
        body = body_content.group(1)
        return [name for (_, name) in ENUM_MEMBER_REGEXP.findall(body)]

    def test_all_names_in_features_lists_exist(self) -> None:
        if False:
            return 10
        missing_names = []
        for feature in self.all_features_set:
            if feature.value not in registry.Registry.parameter_registry:
                missing_names.append(feature.value)
        self.assertTrue(len(missing_names) == 0, msg='Following entries in feature lists are not defined: %s.' % missing_names)

    def test_no_duplicated_names_in_features_lists(self) -> None:
        if False:
            i = 10
            return i + 15
        duplicate_names = []
        for feature in self.all_features_set:
            if self.all_features_list.count(feature) > 1:
                duplicate_names.append(feature.value)
        self.assertTrue(len(duplicate_names) == 0, msg='Following entries appear more than once in features lists: %s.' % duplicate_names)

    def test_no_duplicate_names_in_deprecated_names_list(self) -> None:
        if False:
            i = 10
            return i + 15
        duplicate_names = []
        deprecated_features = platform_feature_list.DEPRECATED_FEATURE_NAMES
        for feature in set(deprecated_features):
            if deprecated_features.count(feature) > 1:
                duplicate_names.append(feature.value)
        self.assertTrue(len(duplicate_names) == 0, msg='Following entries appear more than once in deprecated name list: %s.' % duplicate_names)

    def test_no_deprecated_names_in_features_lists(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        deprecated_names_set = set(platform_feature_list.DEPRECATED_FEATURE_NAMES)
        found_deprecated_names = []
        for feature in self.all_features_set:
            if feature in deprecated_names_set:
                found_deprecated_names.append(feature.value)
        self.assertTrue(len(found_deprecated_names) == 0, msg='Following names in feature lists are deprecated and should not be used: %s.' % found_deprecated_names)

    def test_all_entries_in_features_lists_are_features(self) -> None:
        if False:
            print('Hello World!')
        non_feature_names = []
        for feature in self.all_features_set:
            feature_flag = registry.Registry.get_platform_parameter(feature.value)
            if not feature_flag.is_feature:
                non_feature_names.append(feature.value)
        self.assertTrue(len(non_feature_names) == 0, msg='Following entries in FEATURES_LIST are not features: %s.' % non_feature_names)

    def test_all_entries_in_dev_features_list_are_in_dev_stage(self) -> None:
        if False:
            return 10
        invalid_feature_names = []
        for feature in platform_feature_list.DEV_FEATURES_LIST:
            feature_flag = registry.Registry.get_platform_parameter(feature.value)
            if feature_flag.feature_stage != platform_parameter_domain.FeatureStages.DEV.value:
                invalid_feature_names.append(feature.value)
        self.assertTrue(len(invalid_feature_names) == 0, msg="Following entries defined in DEV_FEATURES_LIST are not in 'dev' stage: %s." % invalid_feature_names)

    def test_all_entries_in_test_features_list_are_in_test_stage(self) -> None:
        if False:
            print('Hello World!')
        invalid_feature_names = []
        for feature in platform_feature_list.TEST_FEATURES_LIST:
            feature_flag = registry.Registry.get_platform_parameter(feature.value)
            if feature_flag.feature_stage != platform_parameter_domain.FeatureStages.TEST.value:
                invalid_feature_names.append(feature.value)
        self.assertTrue(len(invalid_feature_names) == 0, msg="Following entries defined in TEST_FEATURES_LIST are not in 'test' stage: %s." % invalid_feature_names)

    def test_all_entries_in_prod_features_list_are_in_prod_stage(self) -> None:
        if False:
            while True:
                i = 10
        invalid_feature_names = []
        for feature in platform_feature_list.PROD_FEATURES_LIST:
            feature_flag = registry.Registry.get_platform_parameter(feature.value)
            if feature_flag.feature_stage != platform_parameter_domain.FeatureStages.PROD.value:
                invalid_feature_names.append(feature.value)
        self.assertTrue(len(invalid_feature_names) == 0, msg="Following entries defined in PROD_FEATURES_LIST are not in 'prod' stage: %s." % invalid_feature_names)

    def test_all_names_in_features_lists_exist_in_frontend(self) -> None:
        if False:
            print('Hello World!')
        feature_names_in_frontend = self._parse_feature_names_in_frontend()
        all_feature_names_set = [feature.value for feature in self.all_features_set]
        missing_features = set(all_feature_names_set) - set(feature_names_in_frontend)
        self.assertTrue(len(missing_features) == 0, msg='Following entries are not defined in frontend: %s.' % list(missing_features))

    def test_all_names_in_frontend_are_known(self) -> None:
        if False:
            return 10
        feature_names_in_frontend = self._parse_feature_names_in_frontend()
        all_feature_names_set = [feature.value for feature in self.all_features_set]
        missing_features = set(feature_names_in_frontend) - set(all_feature_names_set)
        self.assertTrue(len(missing_features) == 0, msg='Following entries are defined in frontend but not defined in the backend feature list: %s.' % list(missing_features))