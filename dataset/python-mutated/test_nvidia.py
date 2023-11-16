from unittest import mock, TestCase
from golem.envs.docker.vendor import nvidia

class TestIsSupported(TestCase):

    @mock.patch('golem.envs.docker.vendor.nvidia.nvgpu')
    def test_delegated_call(self, nvgpu):
        if False:
            i = 10
            return i + 15
        nvidia.is_supported()
        self.assertEqual(nvgpu.is_supported.call_count, 1)

class TestValidateDevicesFailure(TestCase):

    def test_with_no_devices(self):
        if False:
            return 10
        devices = []
        with self.assertRaises(ValueError) as raised:
            nvidia.validate_devices(devices)
        self.assertEqual(f'Missing {nvidia.VENDOR} GPUs: {devices}', str(raised.exception))

    def test_with_multiple_special_devices(self):
        if False:
            for i in range(10):
                print('nop')
        devices = ['all', 'none']
        with self.assertRaises(ValueError) as raised:
            nvidia.validate_devices(devices)
        self.assertEqual(f'Mixed {nvidia.VENDOR} GPU devices: {devices}', str(raised.exception))

    def test_with_special_plus_other_devices(self):
        if False:
            i = 10
            return i + 15
        devices = ['all', '0', '1']
        with self.assertRaises(ValueError) as raised:
            nvidia.validate_devices(devices)
        self.assertEqual(f'Mixed {nvidia.VENDOR} GPU devices: {devices}', str(raised.exception))

    def test_with_invalid_device_names(self):
        if False:
            for i in range(10):
                print('nop')
        devices = ['/dev/nvidia0', '1']
        with self.assertRaises(ValueError) as raised:
            nvidia.validate_devices(devices)
        self.assertEqual(f'Invalid {nvidia.VENDOR} GPU device names: {devices}', str(raised.exception))

    def test_with_mixed_valid_device_names(self):
        if False:
            for i in range(10):
                print('nop')
        devices = ['GPU-deadbeef-0a0a0a0a-12341234-abcdabcd', '1']
        with self.assertRaises(ValueError) as raised:
            nvidia.validate_devices(devices)
        self.assertEqual(f'Invalid {nvidia.VENDOR} GPU device names: {devices}', str(raised.exception))

class TestValidateDevicesSuccess(TestCase):

    def test_with_index_device_names(self):
        if False:
            while True:
                i = 10
        devices = ['0', '1']
        nvidia.validate_devices(devices)

    def test_with_uuid_device_names(self):
        if False:
            return 10
        devices = ['GPU-deadbeef-0a0a0a0a-00000000-abcdabcd', 'GPU-deadbeef-0a0a0a0a-11111111-aBcDaBcD']
        nvidia.validate_devices(devices)

    def test_special_device_names(self):
        if False:
            return 10
        for special_device in nvidia.SPECIAL_DEVICES:
            nvidia.validate_devices([special_device])

class TestValidateCapabilitiesFailure(TestCase):

    def test_with_no_capabilities(self):
        if False:
            return 10
        caps = []
        with self.assertRaises(ValueError) as raised:
            nvidia.validate_capabilities(caps)
        self.assertEqual(f'Missing {nvidia.VENDOR} GPU caps: {caps}', str(raised.exception))

    def test_with_special_plus_other_capabilities(self):
        if False:
            for i in range(10):
                print('nop')
        caps = [next(iter(nvidia.SPECIAL_CAPABILITIES)), next(iter(nvidia.CAPABILITIES))]
        with self.assertRaises(ValueError) as raised:
            nvidia.validate_capabilities(caps)
        self.assertEqual(f'Mixed {nvidia.VENDOR} GPU caps: {caps}', str(raised.exception))

    def test_with_invalid_capabilities(self):
        if False:
            return 10
        caps = ['_invalid', next(iter(nvidia.CAPABILITIES))]
        with self.assertRaises(ValueError) as raised:
            nvidia.validate_capabilities(caps)
        self.assertEqual(f'Invalid {nvidia.VENDOR} GPU caps: {caps}', str(raised.exception))

class TestValidateCapabilitiesSuccess(TestCase):

    def test_with_special_capabilities(self):
        if False:
            while True:
                i = 10
        for cap in nvidia.SPECIAL_CAPABILITIES:
            nvidia.validate_capabilities([cap])

    def test_with_valid_capabilities(self):
        if False:
            return 10
        for cap in nvidia.CAPABILITIES:
            nvidia.validate_capabilities([cap])

    def test_with_multiple_valid_capabilities(self):
        if False:
            for i in range(10):
                print('nop')
        nvidia.validate_capabilities(list(nvidia.CAPABILITIES))

class TestValidateRequirementsFailure(TestCase):

    def test_invalid_name(self):
        if False:
            i = 10
            return i + 15
        reqs = {'cuda': '>=5.0', '_invalid': '>=6.0'}
        with self.assertRaises(ValueError) as raised:
            nvidia.validate_requirements(reqs)
        self.assertEqual(f"Invalid {nvidia.VENDOR} GPU requirement name: '_invalid'", str(raised.exception))

    def test_missing_value(self):
        if False:
            return 10
        reqs = {'cuda': '>=5.0', 'brand': ''}
        with self.assertRaises(ValueError) as raised:
            nvidia.validate_requirements(reqs)
        self.assertEqual(f"Invalid {nvidia.VENDOR} GPU requirement value: 'brand'=''", str(raised.exception))

class TestValidateRequirementsSuccess(TestCase):

    def test_with_no_requirements(self):
        if False:
            i = 10
            return i + 15
        nvidia.validate_requirements({})

    def test_with_requirements(self):
        if False:
            for i in range(10):
                print('nop')
        nvidia.validate_requirements({'cuda': '>=5.0', 'brand': 'Tesla'})