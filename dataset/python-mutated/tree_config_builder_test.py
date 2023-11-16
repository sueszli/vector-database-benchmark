import unittest
import json
import os
from pokemongo_bot import PokemonGoBot, ConfigException, MismatchTaskApiVersion, TreeConfigBuilder, PluginLoader, BaseTask
from pokemongo_bot.cell_workers import HandleSoftBan, CatchPokemon
from pokemongo_bot.test.resources.plugin_fixture import FakeTask, UnsupportedApiTask

def convert_from_json(str):
    if False:
        while True:
            i = 10
    return json.loads(str)

class TreeConfigBuilderTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.bot = {}

    def test_should_throw_on_no_type_key(self):
        if False:
            return 10
        obj = convert_from_json('[{\n                "bad_key": "foo"\n            }]')
        builder = TreeConfigBuilder(self.bot, obj)
        self.assertRaisesRegexp(ConfigException, 'No type found for given task', builder.build)

    def test_should_throw_on_non_matching_type(self):
        if False:
            for i in range(10):
                print('nop')
        obj = convert_from_json('[{\n                "type": "foo"\n            }]')
        builder = TreeConfigBuilder(self.bot, obj)
        self.assertRaisesRegexp(ConfigException, 'No worker named foo defined', builder.build)

    def test_should_throw_on_wrong_evolve_task_name(self):
        if False:
            print('Hello World!')
        obj = convert_from_json('[{\n                "type": "EvolveAll"\n            }]')
        builder = TreeConfigBuilder(self.bot, obj)
        self.assertRaisesRegexp(ConfigException, 'The EvolveAll task has been renamed to EvolvePokemon', builder.build)

    def test_creating_worker(self):
        if False:
            return 10
        obj = convert_from_json('[{\n                "type": "HandleSoftBan"\n            }]')
        builder = TreeConfigBuilder(self.bot, obj)
        tree = builder.build()
        self.assertIsInstance(tree[0], HandleSoftBan)
        self.assertIs(tree[0].bot, self.bot)

    def test_creating_two_workers(self):
        if False:
            i = 10
            return i + 15
        obj = convert_from_json('[{\n                "type": "HandleSoftBan"\n            }, {\n                "type": "CatchPokemon"\n            }]')
        builder = TreeConfigBuilder(self.bot, obj)
        tree = builder.build()
        self.assertIsInstance(tree[0], HandleSoftBan)
        self.assertIs(tree[0].bot, self.bot)
        self.assertIsInstance(tree[1], CatchPokemon)
        self.assertIs(tree[1].bot, self.bot)

    def test_task_with_config(self):
        if False:
            for i in range(10):
                print('nop')
        obj = convert_from_json('[{\n                "type": "IncubateEggs",\n                "config": {\n                    "longer_eggs_first": true\n                }\n            }]')
        builder = TreeConfigBuilder(self.bot, obj)
        tree = builder.build()
        self.assertTrue(tree[0].config.get('longer_eggs_first', False))

    def test_disabling_task(self):
        if False:
            for i in range(10):
                print('nop')
        obj = convert_from_json('[{\n                "type": "HandleSoftBan",\n                "config": {\n                    "enabled": false\n                }\n            }, {\n                "type": "CatchPokemon",\n                "config": {\n                    "enabled": true\n                }\n            }]')
        builder = TreeConfigBuilder(self.bot, obj)
        tree = builder.build()
        self.assertTrue(len(tree) == 1)
        self.assertIsInstance(tree[0], CatchPokemon)

    def test_load_plugin_task(self):
        if False:
            while True:
                i = 10
        package_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'resources', 'plugin_fixture')
        plugin_loader = PluginLoader()
        plugin_loader.load_plugin(package_path)
        obj = convert_from_json('[{\n            "type": "plugin_fixture.FakeTask"\n        }]')
        builder = TreeConfigBuilder(self.bot, obj)
        tree = builder.build()
        result = tree[0].work()
        self.assertEqual(result, 'FakeTask')

    def setupUnsupportedBuilder(self):
        if False:
            for i in range(10):
                print('nop')
        package_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'pokemongo_bot', 'test', 'resources', 'plugin_fixture')
        plugin_loader = PluginLoader()
        plugin_loader.load_plugin(package_path)
        obj = convert_from_json('[{\n            "type": "plugin_fixture.UnsupportedApiTask"\n        }]')
        return TreeConfigBuilder(self.bot, obj)

    def test_task_version_too_high(self):
        if False:
            while True:
                i = 10
        builder = self.setupUnsupportedBuilder()
        previous_version = BaseTask.TASK_API_VERSION
        BaseTask.TASK_API_VERSION = 1
        self.assertRaisesRegexp(MismatchTaskApiVersion, 'Task plugin_fixture.UnsupportedApiTask only works with task api version 2, you are currently running version 1. Do you need to update the bot?', builder.build)
        BaseTask.TASK_API_VERSION = previous_version

    def test_task_version_too_low(self):
        if False:
            i = 10
            return i + 15
        builder = self.setupUnsupportedBuilder()
        previous_version = BaseTask.TASK_API_VERSION
        BaseTask.TASK_API_VERSION = 3
        self.assertRaisesRegexp(MismatchTaskApiVersion, 'Task plugin_fixture.UnsupportedApiTask only works with task api version 2, you are currently running version 3. Is there a new version of this task?', builder.build)
        BaseTask.TASK_API_VERSION = previous_version