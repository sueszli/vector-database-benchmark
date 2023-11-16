"""Tests for collection domain objects and methods defined on them."""
from __future__ import annotations
import datetime
from core import feconf
from core import utils
from core.constants import constants
from core.domain import collection_domain
from core.domain import collection_services
from core.tests import test_utils
SAMPLE_YAML_CONTENT = 'category: A category\nlanguage_code: en\nnodes:\n- exploration_id: an_exploration_id\nobjective: An objective\nschema_version: %d\ntags: []\ntitle: A title\n' % feconf.CURRENT_COLLECTION_SCHEMA_VERSION

class CollectionChangeTests(test_utils.GenericTestBase):

    def test_collection_change_object_with_missing_cmd(self) -> None:
        if False:
            return 10
        with self.assertRaisesRegex(utils.ValidationError, 'Missing cmd key in change dict'):
            collection_domain.CollectionChange({'invalid': 'data'})

    def test_collection_change_object_with_invalid_cmd(self) -> None:
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(utils.ValidationError, 'Command invalid is not allowed'):
            collection_domain.CollectionChange({'cmd': 'invalid'})

    def test_collection_change_object_with_missing_attribute_in_cmd(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(utils.ValidationError, 'The following required attributes are missing: exploration_id, new_value'):
            collection_domain.CollectionChange({'cmd': 'edit_collection_node_property', 'property_name': 'category', 'old_value': 'old_value'})

    def test_collection_change_object_with_extra_attribute_in_cmd(self) -> None:
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(utils.ValidationError, 'The following extra attributes are present: invalid'):
            collection_domain.CollectionChange({'cmd': 'edit_collection_node_property', 'exploration_id': 'exploration_id', 'property_name': 'category', 'old_value': 'old_value', 'new_value': 'new_value', 'invalid': 'invalid'})

    def test_collection_change_object_with_invalid_collection_property(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(utils.ValidationError, 'Value for property_name in cmd edit_collection_property: invalid is not allowed'):
            collection_domain.CollectionChange({'cmd': 'edit_collection_property', 'property_name': 'invalid', 'old_value': 'old_value', 'new_value': 'new_value'})

    def test_collection_change_object_with_create_new(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        col_change_object = collection_domain.CollectionChange({'cmd': 'create_new', 'category': 'category', 'title': 'title'})
        self.assertEqual(col_change_object.cmd, 'create_new')
        self.assertEqual(col_change_object.category, 'category')
        self.assertEqual(col_change_object.title, 'title')

    def test_collection_change_object_with_add_collection_node(self) -> None:
        if False:
            return 10
        col_change_object = collection_domain.CollectionChange({'cmd': 'add_collection_node', 'exploration_id': 'exploration_id'})
        self.assertEqual(col_change_object.cmd, 'add_collection_node')
        self.assertEqual(col_change_object.exploration_id, 'exploration_id')

    def test_collection_change_object_with_delete_collection_node(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        col_change_object = collection_domain.CollectionChange({'cmd': 'delete_collection_node', 'exploration_id': 'exploration_id'})
        self.assertEqual(col_change_object.cmd, 'delete_collection_node')
        self.assertEqual(col_change_object.exploration_id, 'exploration_id')

    def test_collection_change_object_with_swap_nodes(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        col_change_object = collection_domain.CollectionChange({'cmd': 'swap_nodes', 'first_index': 'first_index', 'second_index': 'second_index'})
        self.assertEqual(col_change_object.cmd, 'swap_nodes')
        self.assertEqual(col_change_object.first_index, 'first_index')
        self.assertEqual(col_change_object.second_index, 'second_index')

    def test_collection_change_object_with_edit_collection_property(self) -> None:
        if False:
            return 10
        col_change_object = collection_domain.CollectionChange({'cmd': 'edit_collection_property', 'property_name': 'category', 'new_value': 'new_value', 'old_value': 'old_value'})
        self.assertEqual(col_change_object.cmd, 'edit_collection_property')
        self.assertEqual(col_change_object.property_name, 'category')
        self.assertEqual(col_change_object.new_value, 'new_value')
        self.assertEqual(col_change_object.old_value, 'old_value')

    def test_collection_change_object_with_edit_collection_node_property(self) -> None:
        if False:
            i = 10
            return i + 15
        col_change_object = collection_domain.CollectionChange({'cmd': 'edit_collection_node_property', 'exploration_id': 'exploration_id', 'property_name': 'title', 'new_value': 'new_value', 'old_value': 'old_value'})
        self.assertEqual(col_change_object.cmd, 'edit_collection_node_property')
        self.assertEqual(col_change_object.exploration_id, 'exploration_id')
        self.assertEqual(col_change_object.property_name, 'title')
        self.assertEqual(col_change_object.new_value, 'new_value')
        self.assertEqual(col_change_object.old_value, 'old_value')

    def test_collection_change_object_with_migrate_schema_to_latest_version(self) -> None:
        if False:
            while True:
                i = 10
        col_change_object = collection_domain.CollectionChange({'cmd': 'migrate_schema_to_latest_version', 'from_version': 'from_version', 'to_version': 'to_version'})
        self.assertEqual(col_change_object.cmd, 'migrate_schema_to_latest_version')
        self.assertEqual(col_change_object.from_version, 'from_version')
        self.assertEqual(col_change_object.to_version, 'to_version')

    def test_collection_change_object_with_add_collection_skill(self) -> None:
        if False:
            return 10
        col_change_object = collection_domain.CollectionChange({'cmd': 'add_collection_skill', 'name': 'name'})
        self.assertEqual(col_change_object.cmd, 'add_collection_skill')
        self.assertEqual(col_change_object.name, 'name')

    def test_collection_change_object_with_delete_collection_skill(self) -> None:
        if False:
            i = 10
            return i + 15
        col_change_object = collection_domain.CollectionChange({'cmd': 'delete_collection_skill', 'skill_id': 'skill_id'})
        self.assertEqual(col_change_object.cmd, 'delete_collection_skill')
        self.assertEqual(col_change_object.skill_id, 'skill_id')

    def test_collection_change_object_with_add_question_id_to_skill(self) -> None:
        if False:
            print('Hello World!')
        col_change_object = collection_domain.CollectionChange({'cmd': 'add_question_id_to_skill', 'skill_id': 'skill_id', 'question_id': 'question_id'})
        self.assertEqual(col_change_object.cmd, 'add_question_id_to_skill')
        self.assertEqual(col_change_object.skill_id, 'skill_id')
        self.assertEqual(col_change_object.question_id, 'question_id')

    def test_collection_change_object_with_remove_question_id_from_skill(self) -> None:
        if False:
            i = 10
            return i + 15
        col_change_object = collection_domain.CollectionChange({'cmd': 'remove_question_id_from_skill', 'skill_id': 'skill_id', 'question_id': 'question_id'})
        self.assertEqual(col_change_object.cmd, 'remove_question_id_from_skill')
        self.assertEqual(col_change_object.skill_id, 'skill_id')
        self.assertEqual(col_change_object.question_id, 'question_id')

    def test_to_dict(self) -> None:
        if False:
            print('Hello World!')
        col_change_dict = {'cmd': 'remove_question_id_from_skill', 'skill_id': 'skill_id', 'question_id': 'question_id'}
        col_change_object = collection_domain.CollectionChange(col_change_dict)
        self.assertEqual(col_change_object.to_dict(), col_change_dict)

class CollectionDomainUnitTests(test_utils.GenericTestBase):
    """Test the collection domain object."""
    COLLECTION_ID = 'collection_id'
    EXPLORATION_ID = 'exp_id_0'

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        super().setUp()
        self.save_new_valid_collection(self.COLLECTION_ID, 'user@example.com', title='Title', category='Category', objective='Objective', exploration_id=self.EXPLORATION_ID)
        self.collection = collection_services.get_collection_by_id(self.COLLECTION_ID)

    def _assert_validation_error(self, expected_error_substring: str) -> None:
        if False:
            while True:
                i = 10
        'Checks that the collection passes strict validation.'
        with self.assertRaisesRegex(utils.ValidationError, expected_error_substring):
            self.collection.validate()

    def test_initial_validation(self) -> None:
        if False:
            while True:
                i = 10
        'Test validating a new, valid collection.'
        self.collection.validate()

    def test_title_validation(self) -> None:
        if False:
            return 10
        self.collection.title = 0
        self._assert_validation_error('Expected title to be a string')

    def test_category_validation(self) -> None:
        if False:
            i = 10
            return i + 15
        self.collection.category = 0
        self._assert_validation_error('Expected category to be a string')

    def test_objective_validation(self) -> None:
        if False:
            return 10
        self.collection.objective = ''
        self._assert_validation_error('objective must be specified')
        self.collection.objective = 0
        self._assert_validation_error('Expected objective to be a string')

    def test_language_code_validation(self) -> None:
        if False:
            return 10
        self.collection.language_code = ''
        self._assert_validation_error('language must be specified')
        self.collection.language_code = 0
        self._assert_validation_error('Expected language code to be a string')
        self.collection.language_code = 'xz'
        self._assert_validation_error('Invalid language code')

    def test_tags_validation(self) -> None:
        if False:
            i = 10
            return i + 15
        self.collection.tags = 'abc'
        self._assert_validation_error('Expected tags to be a list')
        self.collection.tags = [2, 3]
        self._assert_validation_error('Expected each tag to be a string')
        self.collection.tags = ['', 'tag']
        self._assert_validation_error('Tags should be non-empty')
        self.collection.tags = ['234']
        self._assert_validation_error('Tags should only contain lowercase letters and spaces')
        self.collection.tags = ['   abc']
        self._assert_validation_error('Tags should not start or end with whitespace')
        self.collection.tags = ['abc  def']
        self._assert_validation_error('Adjacent whitespace in tags should be collapsed')
        self.collection.tags = ['abc', 'abc']
        self._assert_validation_error('Expected tags to be unique, but found duplicates')

    def test_schema_version_validation(self) -> None:
        if False:
            i = 10
            return i + 15
        self.collection.schema_version = 'some_schema_version'
        self._assert_validation_error('Expected schema version to be an int')
        self.collection.schema_version = 100
        self._assert_validation_error('Expected schema version to be %s' % feconf.CURRENT_COLLECTION_SCHEMA_VERSION)

    def test_nodes_validation(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.collection.nodes = {}
        self._assert_validation_error('Expected nodes to be a list')
        self.collection.nodes = [collection_domain.CollectionNode.from_dict({'exploration_id': '0'}), collection_domain.CollectionNode.from_dict({'exploration_id': '0'})]
        self._assert_validation_error('There are explorations referenced in the collection more than once.')

    def test_initial_explorations_validation(self) -> None:
        if False:
            i = 10
            return i + 15
        self.collection.nodes = []
        self.collection.validate(strict=False)
        self._assert_validation_error('Expected to have at least 1 exploration in the collection.')

    def test_metadata_validation(self) -> None:
        if False:
            return 10
        self.collection.title = ''
        self.collection.objective = ''
        self.collection.category = ''
        self.collection.nodes = []
        self.collection.validate(strict=False)
        self._assert_validation_error('A title must be specified for the collection.')
        self.collection.title = 'A title'
        self.collection.validate(strict=False)
        self._assert_validation_error('An objective must be specified for the collection.')
        self.collection.objective = 'An objective'
        self.collection.validate(strict=False)
        self._assert_validation_error('A category must be specified for the collection.')
        self.collection.category = 'A category'
        self.collection.validate(strict=False)
        self._assert_validation_error('Expected to have at least 1 exploration in the collection.')
        self.collection.add_node('exp_id_1')
        self.collection.validate(strict=False)
        self.collection.validate(strict=True)

    def test_collection_node_exploration_id_validation(self) -> None:
        if False:
            i = 10
            return i + 15
        collection_node0 = self.collection.get_node('exp_id_0')
        collection_node0.exploration_id = 2
        self._assert_validation_error('Expected exploration ID to be a string')

    def test_is_demo_property(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test the is_demo property.'
        demo = collection_domain.Collection.create_default_collection('0')
        self.assertEqual(demo.is_demo, True)
        notdemo1 = collection_domain.Collection.create_default_collection('a')
        self.assertEqual(notdemo1.is_demo, False)
        notdemo2 = collection_domain.Collection.create_default_collection('abcd')
        self.assertEqual(notdemo2.is_demo, False)

    def test_update_title(self) -> None:
        if False:
            return 10
        'Test update_title.'
        self.assertEqual(self.collection.title, 'Title')
        self.collection.update_title('new title')
        self.assertEqual(self.collection.title, 'new title')

    def test_update_category(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test update_category.'
        self.collection.update_category('new category')
        self.assertEqual(self.collection.category, 'new category')

    def test_update_objective(self) -> None:
        if False:
            return 10
        'Test update_objective.'
        self.collection.update_objective('new objective')
        self.assertEqual(self.collection.objective, 'new objective')

    def test_update_language_code(self) -> None:
        if False:
            print('Hello World!')
        'Test update_language_code.'
        self.collection.update_language_code('en')
        self.assertEqual(self.collection.language_code, 'en')

    def test_update_tags(self) -> None:
        if False:
            return 10
        'Test update_tags.'
        self.assertEqual(self.collection.tags, [])
        self.collection.update_tags(['abc', 'def'])
        self.assertEqual(self.collection.tags, ['abc', 'def'])

    def test_collection_export_import(self) -> None:
        if False:
            print('Hello World!')
        'Test that to_dict and from_dict preserve all data within an\n        collection.\n        '
        self.save_new_valid_exploration('0', 'user@example.com', end_state_name='End')
        collection = collection_domain.Collection.create_default_collection('0', title='title', category='category', objective='objective')
        collection_dict = collection.to_dict()
        collection_from_dict = collection_domain.Collection.from_dict(collection_dict)
        self.assertEqual(collection_from_dict.to_dict(), collection_dict)

    def test_add_delete_swap_nodes(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that add_node, delete_node and swap_nodes fail in the correct\n        situations.\n        '
        collection = collection_domain.Collection.create_default_collection('0')
        self.assertEqual(len(collection.nodes), 0)
        collection.add_node('test_exp')
        self.assertEqual(len(collection.nodes), 1)
        with self.assertRaisesRegex(ValueError, 'Exploration is already part of this collection: test_exp'):
            collection.add_node('test_exp')
        collection.add_node('another_exp')
        self.assertEqual(len(collection.nodes), 2)
        collection.swap_nodes(0, 1)
        self.assertEqual(collection.nodes[0].exploration_id, 'another_exp')
        self.assertEqual(collection.nodes[1].exploration_id, 'test_exp')
        with self.assertRaisesRegex(ValueError, 'Both indices point to the same collection node.'):
            collection.swap_nodes(0, 0)
        collection.delete_node('another_exp')
        self.assertEqual(len(collection.nodes), 1)
        with self.assertRaisesRegex(ValueError, 'Exploration is not part of this collection: another_exp'):
            collection.delete_node('another_exp')
        collection.delete_node('test_exp')
        self.assertEqual(len(collection.nodes), 0)

    def test_update_collection_contents_from_model(self) -> None:
        if False:
            while True:
                i = 10
        versioned_collection_contents: collection_domain.VersionedCollectionDict = {'schema_version': 1, 'collection_contents': {}}
        collection_domain.Collection.update_collection_contents_from_model(versioned_collection_contents, 1)
        self.assertEqual(versioned_collection_contents['schema_version'], 2)
        self.assertEqual(versioned_collection_contents['collection_contents'], {})
        collection_domain.Collection.update_collection_contents_from_model(versioned_collection_contents, 2)
        self.assertEqual(versioned_collection_contents['schema_version'], 3)
        self.assertEqual(versioned_collection_contents['collection_contents'], {})

    def test_update_collection_contents_from_model_with_schema_version_5(self) -> None:
        if False:
            i = 10
            return i + 15
        versioned_collection_contents: collection_domain.VersionedCollectionDict = {'schema_version': 5, 'collection_contents': {'nodes': [{'prerequisite_skill_ids': ['11', '22'], 'acquired_skill_ids': ['33', '44'], 'other_field': 'value1'}, {'prerequisite_skill_ids': ['11', '22'], 'acquired_skill_ids': ['33', '44'], 'other_field': 'value2'}]}}
        collection_domain.Collection.update_collection_contents_from_model(versioned_collection_contents, 5)
        self.assertEqual(versioned_collection_contents['schema_version'], 6)
        self.assertEqual(versioned_collection_contents['collection_contents']['nodes'], [{'other_field': 'value1'}, {'other_field': 'value2'}])

    def test_update_collection_contents_from_model_with_invalid_schema_version(self) -> None:
        if False:
            print('Hello World!')
        versioned_collection_contents: collection_domain.VersionedCollectionDict = {'schema_version': feconf.CURRENT_COLLECTION_SCHEMA_VERSION, 'collection_contents': {}}
        with self.assertRaisesRegex(Exception, 'Collection is version .+ but current collection schema version is %d' % feconf.CURRENT_COLLECTION_SCHEMA_VERSION):
            collection_domain.Collection.update_collection_contents_from_model(versioned_collection_contents, feconf.CURRENT_COLLECTION_SCHEMA_VERSION)

    def test_serialize_and_deserialize_returns_unchanged_collection(self) -> None:
        if False:
            print('Hello World!')
        'Checks that serializing and then deserializing a default collection\n        works as intended by leaving the collection unchanged.\n        '
        self.assertEqual(self.collection.to_dict(), collection_domain.Collection.deserialize(self.collection.serialize()).to_dict())

class ExplorationGraphUnitTests(test_utils.GenericTestBase):
    """Test the general structure of explorations within a collection."""

    def test_initial_explorations(self) -> None:
        if False:
            i = 10
            return i + 15
        'Any exploration without prerequisites should be an initial\n        exploration.\n        '
        collection = collection_domain.Collection.create_default_collection('collection_id')
        self.assertEqual(collection.nodes, [])
        self.assertEqual(collection.first_exploration_id, None)
        collection.add_node('exp_id_0')
        self.assertEqual(collection.first_exploration_id, 'exp_id_0')
        collection.add_node('exp_id_1')
        self.assertEqual(len(collection.nodes), 2)
        self.assertEqual(collection.first_exploration_id, 'exp_id_0')

    def test_next_explorations(self) -> None:
        if False:
            while True:
                i = 10
        'Explorations should be suggested based on their index in the node\n           list.\n        '
        collection = collection_domain.Collection.create_default_collection('collection_id')
        self.assertEqual(collection.get_next_exploration_id([]), None)
        collection.add_node('exp_id_0')
        self.assertEqual(collection.get_next_exploration_id([]), 'exp_id_0')
        self.assertEqual(collection.first_exploration_id, collection.get_next_exploration_id([]))
        self.assertEqual(collection.get_next_exploration_id(['exp_id_0']), None)
        collection.add_node('exp_id_1')
        self.assertEqual(collection.get_next_exploration_id(['exp_id_0']), 'exp_id_1')
        collection.add_node('exp_id_2')
        self.assertEqual(collection.get_next_exploration_id([]), 'exp_id_0')
        self.assertEqual(collection.get_next_exploration_id(['exp_id_0']), 'exp_id_1')
        self.assertEqual(collection.get_next_exploration_id(['exp_id_0', 'exp_id_1']), 'exp_id_2')
        self.assertEqual(collection.get_next_exploration_id(['exp_id_0', 'exp_id_1', 'exp_id_2']), None)

    def test_next_explorations_in_sequence(self) -> None:
        if False:
            while True:
                i = 10
        collection = collection_domain.Collection.create_default_collection('collection_id')
        exploration_id = 'exp_id_0'
        collection.add_node(exploration_id)
        self.assertEqual(collection.get_next_exploration_id_in_sequence(exploration_id), None)
        collection.add_node('exp_id_1')
        collection.add_node('exp_id_2')
        self.assertEqual(collection.get_next_exploration_id_in_sequence(exploration_id), 'exp_id_1')
        self.assertEqual(collection.get_next_exploration_id_in_sequence('exp_id_1'), 'exp_id_2')

    def test_nodes_are_in_playble_order(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        collection = collection_domain.Collection.create_default_collection('collection_id')
        self.assertEqual(collection.nodes, [])
        collection.add_node('exp_id_0')
        collection.add_node('exp_id_1')
        collection.add_node('exp_id_2')
        sorted_nodes = collection.nodes
        expected_explorations_ids = ['exp_id_0', 'exp_id_1', 'exp_id_2']
        observed_exploration_ids = [node.exploration_id for node in sorted_nodes]
        self.assertEqual(expected_explorations_ids, observed_exploration_ids)

    def test_next_explorations_with_invalid_exploration_ids(self) -> None:
        if False:
            return 10
        collection = collection_domain.Collection.create_default_collection('collection_id')
        collection.add_node('exp_id_1')
        self.assertEqual(collection.get_next_exploration_id([]), 'exp_id_1')
        self.assertEqual(collection.get_next_exploration_id(['fake_exp_id']), 'exp_id_1')

class YamlCreationUnitTests(test_utils.GenericTestBase):
    """Test creation of collections from YAML files."""
    COLLECTION_ID = 'a_collection_id'
    EXPLORATION_ID = 'an_exploration_id'

    def test_yaml_import_and_export(self) -> None:
        if False:
            print('Hello World!')
        'Test the from_yaml() and to_yaml() methods.'
        self.save_new_valid_exploration(self.EXPLORATION_ID, 'user@example.com', end_state_name='End')
        collection = collection_domain.Collection.create_default_collection(self.COLLECTION_ID, title='A title', category='A category', objective='An objective')
        collection.add_node(self.EXPLORATION_ID)
        self.assertEqual(len(collection.nodes), 1)
        collection.validate()
        yaml_content = collection.to_yaml()
        self.assertEqual(yaml_content, SAMPLE_YAML_CONTENT)
        collection2 = collection_domain.Collection.from_yaml('collection2', yaml_content)
        self.assertEqual(len(collection2.nodes), 1)
        yaml_content_2 = collection2.to_yaml()
        self.assertEqual(yaml_content_2, yaml_content)
        with self.assertRaisesRegex(utils.InvalidInputException, 'Please ensure that you are uploading a YAML text file, not a zip file. The YAML parser returned the following error: '):
            collection_domain.Collection.from_yaml('collection3', '')

    def test_from_yaml_with_no_schema_version_specified_raises_error(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        collection = collection_domain.Collection(self.COLLECTION_ID, 'title', 'category', 'objective', 'en', [], None, [], 0)
        yaml_content = collection.to_yaml()
        with self.assertRaisesRegex(Exception, 'Invalid YAML file: no schema version specified.'):
            collection_domain.Collection.from_yaml(self.COLLECTION_ID, yaml_content)

    def test_from_yaml_with_invalid_schema_version_raises_error(self) -> None:
        if False:
            while True:
                i = 10
        collection = collection_domain.Collection(self.COLLECTION_ID, 'title', 'category', 'objective', 'en', [], 0, [], 0)
        yaml_content = collection.to_yaml()
        with self.assertRaisesRegex(Exception, 'Sorry, we can only process v1 to .+ collection YAML files at present.'):
            collection_domain.Collection.from_yaml(self.COLLECTION_ID, yaml_content)

class SchemaMigrationMethodsUnitTests(test_utils.GenericTestBase):
    """Tests the presence of appropriate schema migration methods in the
    Collection domain object class.
    """

    def test_correct_collection_contents_schema_conversion_methods_exist(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test that the right collection_contents schema conversion methods\n        exist.\n        '
        current_collection_schema_version = feconf.CURRENT_COLLECTION_SCHEMA_VERSION
        for version_num in range(1, current_collection_schema_version):
            self.assertTrue(hasattr(collection_domain.Collection, '_convert_collection_contents_v%s_dict_to_v%s_dict' % (version_num, version_num + 1)))
        self.assertFalse(hasattr(collection_domain.Collection, '_convert_collection_contents_v%s_dict_to_v%s_dict' % (current_collection_schema_version, current_collection_schema_version + 1)))

    def test_correct_collection_schema_conversion_methods_exist(self) -> None:
        if False:
            return 10
        'Test that the right collection schema conversion methods exist.'
        current_collection_schema_version = feconf.CURRENT_COLLECTION_SCHEMA_VERSION
        for version_num in range(1, current_collection_schema_version):
            self.assertTrue(hasattr(collection_domain.Collection, '_convert_v%s_dict_to_v%s_dict' % (version_num, version_num + 1)))
        self.assertFalse(hasattr(collection_domain.Collection, '_convert_v%s_dict_to_v%s_dict' % (current_collection_schema_version, current_collection_schema_version + 1)))

class SchemaMigrationUnitTests(test_utils.GenericTestBase):
    """Test migration methods for yaml content."""
    YAML_CONTENT_V1 = "category: A category\nnodes:\n- acquired_skills:\n  - Skill1\n  - Skill2\n  exploration_id: Exp1\n  prerequisite_skills: []\n- acquired_skills: []\n  exploration_id: Exp2\n  prerequisite_skills:\n  - Skill1\nobjective: ''\nschema_version: 1\ntitle: A title\n"
    YAML_CONTENT_V2 = "category: A category\nlanguage_code: en\nnodes:\n- acquired_skills:\n  - Skill1\n  - Skill2\n  exploration_id: Exp1\n  prerequisite_skills: []\n- acquired_skills: []\n  exploration_id: Exp2\n  prerequisite_skills:\n  - Skill1\nobjective: ''\nschema_version: 2\ntags: []\ntitle: A title\n"
    YAML_CONTENT_V3 = "category: A category\nlanguage_code: en\nnodes:\n- acquired_skills:\n  - Skill1\n  - Skill2\n  exploration_id: Exp1\n  prerequisite_skills: []\n- acquired_skills: []\n  exploration_id: Exp2\n  prerequisite_skills:\n  - Skill1\nobjective: ''\nschema_version: 2\ntags: []\ntitle: A title\n"
    YAML_CONTENT_V4 = "category: A category\nlanguage_code: en\nnext_skill_id: 2\nnodes:\n- acquired_skill_ids:\n  - skill0\n  - skill1\n  exploration_id: Exp1\n  prerequisite_skill_ids: []\n- acquired_skill_ids: []\n  exploration_id: Exp2\n  prerequisite_skill_ids:\n  - skill0\nobjective: ''\nschema_version: 4\nskills:\n  skill0:\n    name: Skill1\n    question_ids: []\n  skill1:\n    name: Skill2\n    question_ids: []\ntags: []\ntitle: A title\n"
    YAML_CONTENT_V5 = "category: A category\nlanguage_code: en\nnext_skill_index: 2\nnodes:\n- acquired_skill_ids:\n  - skill0\n  - skill1\n  exploration_id: Exp1\n  prerequisite_skill_ids: []\n- acquired_skill_ids: []\n  exploration_id: Exp2\n  prerequisite_skill_ids:\n  - skill0\nobjective: ''\nschema_version: 5\nskills:\n  skill0:\n    name: Skill1\n    question_ids: []\n  skill1:\n    name: Skill2\n    question_ids: []\ntags: []\ntitle: A title\n"
    YAML_CONTENT_V6 = "category: A category\nlanguage_code: en\nnodes:\n- exploration_id: Exp1\n- exploration_id: Exp2\nobjective: ''\nschema_version: 6\ntags: []\ntitle: A title\n"
    _LATEST_YAML_CONTENT = YAML_CONTENT_V6

    def test_load_from_v1(self) -> None:
        if False:
            return 10
        'Test direct loading from a v1 yaml file.'
        self.save_new_valid_exploration('Exp1', 'user@example.com', end_state_name='End')
        collection = collection_domain.Collection.from_yaml('cid', self.YAML_CONTENT_V1)
        self.assertEqual(collection.to_yaml(), self._LATEST_YAML_CONTENT)

    def test_load_from_v2(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test direct loading from a v2 yaml file.'
        self.save_new_valid_exploration('Exp1', 'user@example.com', end_state_name='End')
        collection = collection_domain.Collection.from_yaml('cid', self.YAML_CONTENT_V2)
        self.assertEqual(collection.to_yaml(), self._LATEST_YAML_CONTENT)

    def test_load_from_v3(self) -> None:
        if False:
            return 10
        'Test direct loading from a v3 yaml file.'
        self.save_new_valid_exploration('Exp1', 'user@example.com', end_state_name='End')
        collection = collection_domain.Collection.from_yaml('cid', self.YAML_CONTENT_V3)
        self.assertEqual(collection.to_yaml(), self._LATEST_YAML_CONTENT)

    def test_load_from_v4(self) -> None:
        if False:
            print('Hello World!')
        'Test direct loading from a v4 yaml file.'
        self.save_new_valid_exploration('Exp1', 'user@example.com', end_state_name='End')
        collection = collection_domain.Collection.from_yaml('cid', self.YAML_CONTENT_V4)
        self.assertEqual(collection.to_yaml(), self._LATEST_YAML_CONTENT)

    def test_load_from_v5(self) -> None:
        if False:
            print('Hello World!')
        'Test direct loading from a v5 yaml file.'
        self.save_new_valid_exploration('Exp1', 'user@example.com', end_state_name='End')
        collection = collection_domain.Collection.from_yaml('cid', self.YAML_CONTENT_V5)
        self.assertEqual(collection.to_yaml(), self._LATEST_YAML_CONTENT)

    def test_load_from_v6(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test direct loading from a v6 yaml file.'
        self.save_new_valid_exploration('Exp1', 'user@example.com', end_state_name='End')
        collection = collection_domain.Collection.from_yaml('cid', self.YAML_CONTENT_V6)
        self.assertEqual(collection.to_yaml(), self._LATEST_YAML_CONTENT)

class CollectionSummaryTests(test_utils.GenericTestBase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        super().setUp()
        current_time = datetime.datetime.utcnow()
        self.collection_summary_dict = {'category': 'category', 'status': constants.ACTIVITY_STATUS_PRIVATE, 'community_owned': True, 'viewer_ids': ['viewer_id'], 'version': 1, 'editor_ids': ['editor_id'], 'title': 'title', 'collection_model_created_on': current_time, 'tags': [], 'collection_model_last_updated': current_time, 'contributor_ids': ['contributor_id'], 'language_code': 'en', 'objective': 'objective', 'contributors_summary': {}, 'id': 'col_id', 'owner_ids': ['owner_id']}
        self.collection_summary = collection_domain.CollectionSummary('col_id', 'title', 'category', 'objective', 'en', [], constants.ACTIVITY_STATUS_PRIVATE, True, ['owner_id'], ['editor_id'], ['viewer_id'], ['contributor_id'], {}, 1, 1, current_time, current_time)

    def test_collection_summary_gets_created(self) -> None:
        if False:
            return 10
        self.assertEqual(self.collection_summary.to_dict(), self.collection_summary_dict)

    def test_validation_passes_with_valid_properties(self) -> None:
        if False:
            print('Hello World!')
        self.collection_summary.validate()

    def test_validation_fails_with_unallowed_language_code(self) -> None:
        if False:
            print('Hello World!')
        self.collection_summary.language_code = 'invalid'
        with self.assertRaisesRegex(utils.ValidationError, 'Invalid language code: invalid'):
            self.collection_summary.validate()

    def test_validation_fails_with_empty_tag_in_tags(self) -> None:
        if False:
            while True:
                i = 10
        self.collection_summary.tags = ['', 'abc']
        with self.assertRaisesRegex(utils.ValidationError, 'Tags should be non-empty'):
            self.collection_summary.validate()

    def test_validation_fails_with_unallowed_characters_in_tag(self) -> None:
        if False:
            i = 10
            return i + 15
        self.collection_summary.tags = ['123', 'abc']
        with self.assertRaisesRegex(utils.ValidationError, "Tags should only contain lowercase letters and spaces, received '123'"):
            self.collection_summary.validate()

    def test_validation_fails_with_whitespace_in_tag_start(self) -> None:
        if False:
            i = 10
            return i + 15
        self.collection_summary.tags = [' ab', 'abc']
        with self.assertRaisesRegex(utils.ValidationError, "Tags should not start or end with whitespace, received ' ab'"):
            self.collection_summary.validate()

    def test_validation_fails_with_whitespace_in_tag_end(self) -> None:
        if False:
            return 10
        self.collection_summary.tags = ['ab ', 'abc']
        with self.assertRaisesRegex(utils.ValidationError, "Tags should not start or end with whitespace, received 'ab '"):
            self.collection_summary.validate()

    def test_validation_fails_with_adjacent_whitespace_in_tag(self) -> None:
        if False:
            while True:
                i = 10
        self.collection_summary.tags = ['a   b', 'abc']
        with self.assertRaisesRegex(utils.ValidationError, "Adjacent whitespace in tags should be collapsed, received 'a   b'"):
            self.collection_summary.validate()

    def test_validation_fails_with_duplicate_tags(self) -> None:
        if False:
            i = 10
            return i + 15
        self.collection_summary.tags = ['abc', 'abc', 'ab']
        with self.assertRaisesRegex(utils.ValidationError, 'Expected tags to be unique, but found duplicates'):
            self.collection_summary.validate()

    def test_is_private(self) -> None:
        if False:
            while True:
                i = 10
        self.assertTrue(self.collection_summary.is_private())
        self.collection_summary = collection_domain.CollectionSummary('col_id', 'title', 'category', 'objective', 'en', [], constants.ACTIVITY_STATUS_PUBLIC, True, ['owner_id'], ['editor_id'], ['viewer_id'], ['contributor_id'], {}, 1, 1, datetime.datetime.utcnow(), datetime.datetime.utcnow())
        self.assertFalse(self.collection_summary.is_private())

    def test_is_editable_by(self) -> None:
        if False:
            print('Hello World!')
        self.assertTrue(self.collection_summary.is_editable_by('editor_id'))
        self.assertTrue(self.collection_summary.is_editable_by('other_id'))
        self.collection_summary = collection_domain.CollectionSummary('col_id', 'title', 'category', 'objective', 'en', [], constants.ACTIVITY_STATUS_PUBLIC, False, ['owner_id'], ['editor_id'], ['viewer_id'], ['contributor_id'], {}, 1, 1, datetime.datetime.utcnow(), datetime.datetime.utcnow())
        self.assertFalse(self.collection_summary.is_editable_by('other_id'))

    def test_is_solely_owned_by_user_one_owner(self) -> None:
        if False:
            return 10
        self.assertTrue(self.collection_summary.is_solely_owned_by_user('owner_id'))
        self.assertFalse(self.collection_summary.is_solely_owned_by_user('other_id'))
        self.collection_summary = collection_domain.CollectionSummary('col_id', 'title', 'category', 'objective', 'en', [], constants.ACTIVITY_STATUS_PUBLIC, True, ['other_id'], ['editor_id'], ['viewer_id'], ['contributor_id'], {}, 1, 1, datetime.datetime.utcnow(), datetime.datetime.utcnow())
        self.assertFalse(self.collection_summary.is_solely_owned_by_user('owner_id'))
        self.assertTrue(self.collection_summary.is_solely_owned_by_user('other_id'))

    def test_is_solely_owned_by_user_multiple_owners(self) -> None:
        if False:
            while True:
                i = 10
        self.assertTrue(self.collection_summary.is_solely_owned_by_user('owner_id'))
        self.assertFalse(self.collection_summary.is_solely_owned_by_user('other_id'))
        self.collection_summary = collection_domain.CollectionSummary('col_id', 'title', 'category', 'objective', 'en', [], constants.ACTIVITY_STATUS_PUBLIC, True, ['owner_id', 'other_id'], ['editor_id'], ['viewer_id'], ['contributor_id'], {}, 1, 1, datetime.datetime.utcnow(), datetime.datetime.utcnow())
        self.assertFalse(self.collection_summary.is_solely_owned_by_user('owner_id'))
        self.assertFalse(self.collection_summary.is_solely_owned_by_user('other_id'))

    def test_is_solely_owned_by_user_other_users(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertFalse(self.collection_summary.is_solely_owned_by_user('editor_id'))
        self.assertFalse(self.collection_summary.is_solely_owned_by_user('viewer_id'))
        self.assertFalse(self.collection_summary.is_solely_owned_by_user('contributor_id'))

    def test_does_user_have_any_role(self) -> None:
        if False:
            while True:
                i = 10
        self.assertTrue(self.collection_summary.does_user_have_any_role('owner_id'))
        self.assertTrue(self.collection_summary.does_user_have_any_role('viewer_id'))
        self.assertFalse(self.collection_summary.does_user_have_any_role('other_id'))

    def test_add_new_contribution_for_user_adds_user_to_contributors(self) -> None:
        if False:
            i = 10
            return i + 15
        self.collection_summary.add_contribution_by_user('user_id')
        self.assertIn('user_id', self.collection_summary.contributors_summary)
        self.assertEqual(self.collection_summary.contributors_summary['user_id'], 1)
        self.assertIn('user_id', self.collection_summary.contributor_ids)

    def test_add_new_contribution_for_user_increases_score_in_contributors(self) -> None:
        if False:
            return 10
        self.collection_summary.add_contribution_by_user('user_id')
        self.collection_summary.add_contribution_by_user('user_id')
        self.assertIn('user_id', self.collection_summary.contributors_summary)
        self.assertEqual(self.collection_summary.contributors_summary['user_id'], 2)

    def test_add_new_contribution_for_user_does_not_add_system_user(self) -> None:
        if False:
            while True:
                i = 10
        self.collection_summary.add_contribution_by_user(feconf.SYSTEM_COMMITTER_ID)
        self.assertNotIn(feconf.SYSTEM_COMMITTER_ID, self.collection_summary.contributors_summary)
        self.assertNotIn(feconf.SYSTEM_COMMITTER_ID, self.collection_summary.contributor_ids)