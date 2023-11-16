"""Unit tests for scripts/extend_index_yaml.py."""
from __future__ import annotations
import tempfile
from core.tests import test_utils
from . import extend_index_yaml

class ReformatXmlToYamlTests(test_utils.GenericTestBase):
    """Class for testing the reformat_xml_dict_into_yaml_dict function."""

    def test_dict_with_one_index_one_attribute_ascending(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        xml_dict: extend_index_yaml.XmlIndexesDict = {'datastore-indexes': {'datastore-index': [{'@kind': 'TopicModel', '@ancestor': 'false', '@source-service': 'auto', 'property': [{'@name': 'property', '@direction': 'asc'}]}]}}
        expected_yaml_dict: extend_index_yaml.YamlIndexesDict = {'indexes': [{'kind': 'TopicModel', 'properties': [{'name': 'property'}]}]}
        self.assertEqual(extend_index_yaml.reformat_xml_dict_into_yaml_dict(xml_dict), expected_yaml_dict)

    def test_dict_with_one_index_multiple_attributes_ascending(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        xml_dict: extend_index_yaml.XmlIndexesDict = {'datastore-indexes': {'datastore-index': [{'@kind': 'TopicModel', '@ancestor': 'false', '@source-service': 'auto', 'property': [{'@name': 'property1', '@direction': 'asc'}, {'@name': 'property2', '@direction': 'asc'}]}]}}
        expected_yaml_dict: extend_index_yaml.YamlIndexesDict = {'indexes': [{'kind': 'TopicModel', 'properties': [{'name': 'property1'}, {'name': 'property2'}]}]}
        self.assertEqual(extend_index_yaml.reformat_xml_dict_into_yaml_dict(xml_dict), expected_yaml_dict)

    def test_dict_with_multiple_indexes_properties_descending(self) -> None:
        if False:
            print('Hello World!')
        xml_dict: extend_index_yaml.XmlIndexesDict = {'datastore-indexes': {'datastore-index': [{'@kind': 'TopicModel', '@ancestor': 'false', '@source-service': 'auto', 'property': [{'@name': 'property1', '@direction': 'asc'}, {'@name': 'property2', '@direction': 'desc'}]}, {'@kind': 'CollectionModel', '@ancestor': 'false', '@source-service': 'auto', 'property': [{'@name': 'property3', '@direction': 'asc'}, {'@name': 'property4', '@direction': 'desc'}]}]}}
        expected_yaml_dict: extend_index_yaml.YamlIndexesDict = {'indexes': [{'kind': 'TopicModel', 'properties': [{'name': 'property1'}, {'name': 'property2', 'direction': 'desc'}]}, {'kind': 'CollectionModel', 'properties': [{'name': 'property3'}, {'name': 'property4', 'direction': 'desc'}]}]}
        self.assertEqual(extend_index_yaml.reformat_xml_dict_into_yaml_dict(xml_dict), expected_yaml_dict)

class ExtendIndexYamlTests(test_utils.GenericTestBase):
    """Class for testing the extend_index_yaml script."""

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        super().setUp()
        self.index_yaml_file = tempfile.NamedTemporaryFile()
        self.web_inf_index_xml_file = tempfile.NamedTemporaryFile()
        self.index_yaml_file_name = self.index_yaml_file.name
        self.web_inf_index_xml_file_name = self.web_inf_index_xml_file.name
        self.index_yaml_swap = self.swap(extend_index_yaml, 'INDEX_YAML_PATH', self.index_yaml_file.name)
        self.web_inf_index_xml_swap = self.swap(extend_index_yaml, 'WEB_INF_INDEX_XML_PATH', self.web_inf_index_xml_file.name)
        self.open_index_yaml_r = open(self.index_yaml_file.name, 'r', encoding='utf-8')
        self.open_index_yaml_w = open(self.index_yaml_file.name, 'w', encoding='utf-8')
        self.open_web_inf_index_xml = open(self.web_inf_index_xml_file.name, 'a', encoding='utf-8')

    def _run_test_for_extend_index_yaml(self, index_yaml: str, web_inf_index_xml: str, expected_index_yaml: str) -> None:
        if False:
            i = 10
            return i + 15
        'Run tests for extend_index_yaml script.'
        with self.index_yaml_swap, self.web_inf_index_xml_swap:
            with self.open_index_yaml_w as f:
                f.write(index_yaml)
            with self.open_web_inf_index_xml as f:
                f.write(web_inf_index_xml)
            extend_index_yaml.main()
            with self.open_index_yaml_r as f:
                actual_index_yaml = f.read()
            self.assertEqual(actual_index_yaml, expected_index_yaml)

    def tearDown(self) -> None:
        if False:
            return 10
        super().tearDown()
        self.index_yaml_file.close()
        self.web_inf_index_xml_file.close()

    def test_extend_index_yaml_with_changes(self) -> None:
        if False:
            while True:
                i = 10
        index_yaml = 'indexes:\n\n- kind: AppFeedbackReportModel\n  properties:\n  - name: created_on\n  - name: scrubbed_by\n\n- kind: BlogPostRightsModel\n  properties:\n  - name: blog_post_is_published\n  - name: editor_ids\n  - name: last_updated\n    direction: desc\n'
        web_inf_index_xml = '\n<datastore-indexes autoGenerate="true">    \n    <datastore-index kind="ClassifierTrainingJobModel" ancestor="false" source="auto">\n        <property name="status" direction="asc"/>\n        <property name="next_scheduled_check_time" direction="asc"/>\n    </datastore-index>\n</datastore-indexes>\n'
        expected_index_yaml = 'indexes:\n\n- kind: AppFeedbackReportModel\n  properties:\n  - name: created_on\n  - name: scrubbed_by\n\n- kind: BlogPostRightsModel\n  properties:\n  - name: blog_post_is_published\n  - name: editor_ids\n  - name: last_updated\n    direction: desc\n\n- kind: ClassifierTrainingJobModel\n  properties:\n  - name: status\n  - name: next_scheduled_check_time\n'
        self._run_test_for_extend_index_yaml(index_yaml, web_inf_index_xml, expected_index_yaml)

    def test_extend_index_yaml_without_changes(self) -> None:
        if False:
            while True:
                i = 10
        index_yaml = 'indexes:\n\n- kind: AppFeedbackReportModel\n  properties:\n  - name: created_on\n  - name: scrubbed_by\n\n- kind: BlogPostRightsModel\n  properties:\n  - name: blog_post_is_published\n  - name: editor_ids\n  - name: last_updated\n    direction: desc\n'
        web_inf_index_xml = '\n<datastore-indexes autoGenerate="true">    \n    <datastore-index kind="BlogPostRightsModel" ancestor="false" source="auto">\n        <property name="blog_post_is_published" direction="asc"/>\n        <property name="editor_ids" direction="asc"/>\n        <property name="last_updated" direction="desc"/>\n    </datastore-index>\n</datastore-indexes>\n'
        self._run_test_for_extend_index_yaml(index_yaml, web_inf_index_xml, index_yaml)

    def test_extend_index_yaml_with_empty_web_inf_ind_xml(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        index_yaml = 'indexes:\n\n- kind: AppFeedbackReportModel\n  properties:\n  - name: created_on\n  - name: scrubbed_by\n\n- kind: BlogPostRightsModel\n  properties:\n  - name: blog_post_is_published\n  - name: editor_ids\n  - name: last_updated\n    direction: desc\n'
        web_inf_index_xml = '\n<datastore-indexes autoGenerate="true"/>\n'
        self._run_test_for_extend_index_yaml(index_yaml, web_inf_index_xml, index_yaml)

    def test_extend_index_yaml_with_same_kind(self) -> None:
        if False:
            print('Hello World!')
        index_yaml = 'indexes:\n\n- kind: AppFeedbackReportModel\n  properties:\n  - name: created_on\n  - name: scrubbed_by\n\n- kind: BlogPostRightsModel\n  properties:\n  - name: blog_post_is_published\n  - name: editor_ids\n  - name: last_updated\n    direction: desc\n\n- kind: ClassifierTrainingJobModel\n  properties:\n  - name: task\n  - name: status\n  - name: next_scheduled_check_time\n'
        web_inf_index_xml = '\n<datastore-indexes autoGenerate="true">    \n    <datastore-index kind="ClassifierTrainingJobModel" ancestor="false" source="auto">\n        <property name="status" direction="asc"/>\n        <property name="next_scheduled_check_time" direction="asc"/>\n    </datastore-index>\n</datastore-indexes>\n'
        expected_index_yaml = 'indexes:\n\n- kind: AppFeedbackReportModel\n  properties:\n  - name: created_on\n  - name: scrubbed_by\n\n- kind: BlogPostRightsModel\n  properties:\n  - name: blog_post_is_published\n  - name: editor_ids\n  - name: last_updated\n    direction: desc\n\n- kind: ClassifierTrainingJobModel\n  properties:\n  - name: task\n  - name: status\n  - name: next_scheduled_check_time\n\n- kind: ClassifierTrainingJobModel\n  properties:\n  - name: status\n  - name: next_scheduled_check_time\n'
        self._run_test_for_extend_index_yaml(index_yaml, web_inf_index_xml, expected_index_yaml)

    def test_extend_index_yaml_with_same_kind_in_web_inf_xml(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        index_yaml = 'indexes:\n\n- kind: AppFeedbackReportModel\n  properties:\n  - name: created_on\n  - name: scrubbed_by\n\n- kind: BlogPostRightsModel\n  properties:\n  - name: blog_post_is_published\n  - name: editor_ids\n  - name: last_updated\n    direction: desc\n'
        web_inf_index_xml = '\n<datastore-indexes autoGenerate="true">    \n    <datastore-index kind="ClassifierTrainingJobModel" ancestor="false" source="auto">\n        <property name="task" direction="asc"/>\n        <property name="status" direction="asc"/>\n        <property name="next_scheduled_check_time" direction="asc"/>\n    </datastore-index>\n    <datastore-index kind="ClassifierTrainingJobModel" ancestor="false" source="auto">\n        <property name="status" direction="asc"/>\n        <property name="next_scheduled_check_time" direction="asc"/>\n    </datastore-index>\n</datastore-indexes>\n'
        expected_index_yaml = 'indexes:\n\n- kind: AppFeedbackReportModel\n  properties:\n  - name: created_on\n  - name: scrubbed_by\n\n- kind: BlogPostRightsModel\n  properties:\n  - name: blog_post_is_published\n  - name: editor_ids\n  - name: last_updated\n    direction: desc\n\n- kind: ClassifierTrainingJobModel\n  properties:\n  - name: task\n  - name: status\n  - name: next_scheduled_check_time\n\n- kind: ClassifierTrainingJobModel\n  properties:\n  - name: status\n  - name: next_scheduled_check_time\n'
        self._run_test_for_extend_index_yaml(index_yaml, web_inf_index_xml, expected_index_yaml)

    def test_extend_index_yaml_with_same_kind_different_order(self) -> None:
        if False:
            return 10
        index_yaml = 'indexes:\n\n- kind: AppFeedbackReportModel\n  properties:\n  - name: created_on\n  - name: scrubbed_by\n\n- kind: BlogPostRightsModel\n  properties:\n  - name: blog_post_is_published\n  - name: editor_ids\n  - name: story_ids\n  - name: last_updated\n    direction: desc\n'
        web_inf_index_xml = '\n<datastore-indexes autoGenerate="true">  \n    <datastore-index kind="BlogPostRightsModel" ancestor="false" source="auto">\n        <property name="editor_ids" direction="asc"/>\n        <property name="blog_post_is_published" direction="asc"/>\n        <property name="story_ids2" direction="asc"/>\n        <property name="last_updated" direction="desc"/>\n    </datastore-index>  \n    <datastore-index kind="ClassifierTrainingJobModel" ancestor="false" source="auto">\n        <property name="task" direction="asc"/>\n        <property name="status" direction="asc"/>\n        <property name="next_scheduled_check_time" direction="asc"/>\n    </datastore-index>\n    <datastore-index kind="ClassifierTrainingJobModel" ancestor="false" source="auto">\n        <property name="status" direction="asc"/>\n        <property name="next_scheduled_check_time" direction="asc"/>\n    </datastore-index>\n</datastore-indexes>\n'
        expected_index_yaml = 'indexes:\n\n- kind: AppFeedbackReportModel\n  properties:\n  - name: created_on\n  - name: scrubbed_by\n\n- kind: BlogPostRightsModel\n  properties:\n  - name: blog_post_is_published\n  - name: editor_ids\n  - name: story_ids\n  - name: last_updated\n    direction: desc\n\n- kind: BlogPostRightsModel\n  properties:\n  - name: editor_ids\n  - name: blog_post_is_published\n  - name: story_ids2\n  - name: last_updated\n    direction: desc\n\n- kind: ClassifierTrainingJobModel\n  properties:\n  - name: task\n  - name: status\n  - name: next_scheduled_check_time\n\n- kind: ClassifierTrainingJobModel\n  properties:\n  - name: status\n  - name: next_scheduled_check_time\n'
        self._run_test_for_extend_index_yaml(index_yaml, web_inf_index_xml, expected_index_yaml)