import uuid
from peewee import DoesNotExist, IntegrityError
from golem.appconfig import DEFAULT_HARDWARE_PRESET_NAME, CUSTOM_HARDWARE_PRESET_NAME
from golem.hardware.presets import HardwarePresets, HardwarePresetsMixin
from golem.model import HardwarePreset
from golem.tools.testwithdatabase import TestWithDatabase

class TestHardwarePresetsMixin(TestWithDatabase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(TestHardwarePresetsMixin, self).setUp()

    @staticmethod
    def create_sample_preset_dict():
        if False:
            while True:
                i = 10
        return {'name': str(uuid.uuid4()), 'cpu_cores': 1, 'memory': 1000 * 1024, 'disk': 1000 * 1024}

    def test_get_hw_presets(self):
        if False:
            print('Hello World!')
        HardwarePresets.initialize(self.tempdir)
        presets = HardwarePresetsMixin.get_hw_presets()
        assert len(presets) >= 2
        assert all([preset is not None for preset in presets])

    def test_get_hw_preset(self):
        if False:
            print('Hello World!')
        HardwarePresets.initialize(self.tempdir)
        assert HardwarePresetsMixin.get_hw_preset(DEFAULT_HARDWARE_PRESET_NAME)
        assert HardwarePresetsMixin.get_hw_preset(CUSTOM_HARDWARE_PRESET_NAME)
        with self.assertRaises(DoesNotExist):
            assert not HardwarePresetsMixin.get_hw_preset(str(uuid.uuid4()))

    def test_create_hw_preset(self):
        if False:
            print('Hello World!')
        preset_name = str(uuid.uuid4())
        preset_cpu_cores = 1
        preset_memory = 1000 * 1024
        preset_disk = 1000 * 1024
        preset_dict = dict()
        with self.assertRaises(IntegrityError):
            HardwarePresetsMixin.create_hw_preset(preset_dict)
        preset_dict['name'] = preset_name
        with self.assertRaises(IntegrityError):
            HardwarePresetsMixin.create_hw_preset(preset_dict)
        preset_dict['cpu_cores'] = preset_cpu_cores
        with self.assertRaises(IntegrityError):
            HardwarePresetsMixin.create_hw_preset(preset_dict)
        preset_dict['memory'] = preset_memory
        with self.assertRaises(IntegrityError):
            HardwarePresetsMixin.create_hw_preset(preset_dict)
        preset_dict['disk'] = preset_disk
        assert HardwarePresetsMixin.create_hw_preset(preset_dict)
        preset = HardwarePresetsMixin.get_hw_preset(preset_name)
        with self.assertRaises(IntegrityError):
            HardwarePresetsMixin.create_hw_preset(preset_dict)
        assert preset
        assert preset['name'] == preset_name
        assert preset['cpu_cores'] == preset_cpu_cores
        assert preset['memory'] == preset_memory
        assert preset['disk'] == preset_disk
        preset_dict['name'] = str(uuid.uuid4())
        print(preset_dict)
        assert HardwarePresetsMixin.upsert_hw_preset(preset_dict)
        assert HardwarePresetsMixin.get_hw_preset(preset_dict['name'])
        preset_dict['name'] = str(uuid.uuid4())
        preset = HardwarePreset(**preset_dict)
        assert HardwarePresetsMixin.upsert_hw_preset(preset)
        assert HardwarePresetsMixin.get_hw_preset(preset_dict['name'])

    def test_update_hw_preset(self):
        if False:
            print('Hello World!')
        preset_dict = self.create_sample_preset_dict()
        assert HardwarePresetsMixin.create_hw_preset(preset_dict)
        preset_dict['cpu_cores'] += 1
        assert HardwarePresetsMixin.update_hw_preset(preset_dict)
        preset = HardwarePresetsMixin.get_hw_preset(preset_dict['name'])
        assert preset['cpu_cores'] == preset_dict['cpu_cores']
        preset_dict['cpu_cores'] += 1
        preset = HardwarePresetsMixin.upsert_hw_preset(preset_dict)
        assert preset['cpu_cores'] == preset_dict['cpu_cores']

    def test_delete_hw_preset(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            HardwarePresetsMixin.delete_hw_preset(DEFAULT_HARDWARE_PRESET_NAME)
        with self.assertRaises(ValueError):
            HardwarePresetsMixin.delete_hw_preset(CUSTOM_HARDWARE_PRESET_NAME)
        assert not HardwarePresetsMixin.delete_hw_preset(str(uuid.uuid4()))
        preset_dict = self.create_sample_preset_dict()
        assert HardwarePresetsMixin.create_hw_preset(preset_dict)
        assert HardwarePresetsMixin.delete_hw_preset(preset_dict['name'])
        assert not HardwarePresetsMixin.delete_hw_preset(preset_dict['name'])

    def test_sanitize_preset_name(self):
        if False:
            i = 10
            return i + 15
        sanitize = HardwarePresetsMixin._HardwarePresetsMixin__sanitize_preset_name
        assert sanitize(None) == CUSTOM_HARDWARE_PRESET_NAME
        assert sanitize('') == CUSTOM_HARDWARE_PRESET_NAME
        assert sanitize(DEFAULT_HARDWARE_PRESET_NAME) == CUSTOM_HARDWARE_PRESET_NAME
        assert sanitize(CUSTOM_HARDWARE_PRESET_NAME) == CUSTOM_HARDWARE_PRESET_NAME
        assert sanitize('test') == 'test'