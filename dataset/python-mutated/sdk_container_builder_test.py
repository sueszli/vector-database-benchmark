"""Unit tests for the sdk container builder module."""
import gc
import logging
import unittest
import unittest.mock
from apache_beam.options import pipeline_options
from apache_beam.runners.portability import sdk_container_builder

class SdkContainerBuilderTest(unittest.TestCase):

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        gc.collect()

    def test_can_find_local_builder(self):
        if False:
            return 10
        local_builder = sdk_container_builder.SdkContainerImageBuilder._get_subclass_by_key('local_docker')
        self.assertEqual(local_builder, sdk_container_builder._SdkContainerImageLocalBuilder)

    def test_can_find_cloud_builder(self):
        if False:
            return 10
        local_builder = sdk_container_builder.SdkContainerImageBuilder._get_subclass_by_key('cloud_build')
        self.assertEqual(local_builder, sdk_container_builder._SdkContainerImageCloudBuilder)

    def test_missing_builder_key_throws_value_error(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError):
            sdk_container_builder.SdkContainerImageBuilder._get_subclass_by_key('missing-id')

    def test_multiple_matchings_keys_throws_value_error(self):
        if False:
            for i in range(10):
                print('nop')

        class _PluginSdkBuilder(sdk_container_builder.SdkContainerImageBuilder):

            @classmethod
            def _builder_key(cls):
                if False:
                    for i in range(10):
                        print('nop')
                return 'test-id'

        class _PluginSdkBuilder2(sdk_container_builder.SdkContainerImageBuilder):

            @classmethod
            def _builder_key(cls):
                if False:
                    i = 10
                    return i + 15
                return 'test-id'
        with self.assertRaises(ValueError):
            sdk_container_builder.SdkContainerImageBuilder._get_subclass_by_key('test-id')

    def test_can_find_new_subclass(self):
        if False:
            print('Hello World!')

        class _PluginSdkBuilder(sdk_container_builder.SdkContainerImageBuilder):
            pass
        expected_key = f'{_PluginSdkBuilder.__module__}._PluginSdkBuilder'
        local_builder = sdk_container_builder.SdkContainerImageBuilder._get_subclass_by_key(expected_key)
        self.assertEqual(local_builder, _PluginSdkBuilder)

    @unittest.mock.patch('apache_beam.runners.portability.sdk_container_builder._SdkContainerImageLocalBuilder')
    @unittest.mock.patch.object(sdk_container_builder.SdkContainerImageBuilder, '_get_subclass_by_key')
    def test_build_container_image_locates_subclass_invokes_build(self, mock_get_subclass, mocked_local_builder):
        if False:
            return 10
        mock_get_subclass.return_value = mocked_local_builder
        options = pipeline_options.PipelineOptions()
        sdk_container_builder.SdkContainerImageBuilder.build_container_image(options)
        mocked_local_builder.assert_called_once_with(options)
        mocked_local_builder.return_value._build.assert_called_once_with()
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()