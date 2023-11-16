import os
import tempfile
import unittest
from tools.gen_vulkan_spv import VulkanShaderGenerator
from yaml.constructor import ConstructorError

class TestVulkanShaderCodegen(unittest.TestCase):

    def test_assert_on_duplicate_key_yaml(self) -> None:
        if False:
            print('Hello World!')
        yaml_with_duplicate_keys = '\nconv2d_pw:\n  parameter_names_with_default_values:\n      NAME: conv2d_pw_1x1\n      TILE_SIZE_X: 1\n      TILE_SIZE_Y: 1\n  parameter_values:\n    - NAME: conv2d_pw_2x2\n      TILE_SIZE_X: 2\n      TILE_SIZE_Y: 2\n    - NAME: conv2d_pw_2x4\n      TILE_SIZE_X: 2\n      TILE_SIZE_Y: 4\n    - NAME: conv2d_pw_4x2\n      TILE_SIZE_X: 4\n      TILE_SIZE_Y: 2\n    - NAME: conv2d_pw_4x4\n      TILE_SIZE_X: 4\n      TILE_SIZE_Y: 4\nconv2d_pw:\n  parameter_names_with_default_values:\n      NAME: conv2d_pw_1x1\n      TILE_SIZE_X: 1\n      TILE_SIZE_Y: 1\n  parameter_values:\n    - NAME: conv2d_pw_2x2\n      TILE_SIZE_X: 2\n      TILE_SIZE_Y: 2\n    - NAME: conv2d_pw_2x4\n      TILE_SIZE_X: 2\n      TILE_SIZE_Y: 4\n    - NAME: conv2d_pw_4x2\n      TILE_SIZE_X: 4\n      TILE_SIZE_Y: 2\n    - NAME: conv2d_pw_4x4\n      TILE_SIZE_X: 4\n      TILE_SIZE_Y: 4\n'
        generator = VulkanShaderGenerator()
        with tempfile.NamedTemporaryFile(mode='w') as fp:
            fp.write(yaml_with_duplicate_keys)
            fp.flush()
            with self.assertRaisesRegex(ConstructorError, 'while constructing a mapping'):
                generator.add_params_yaml(fp.name)

    def test_assert_keys_mismatch(self) -> None:
        if False:
            print('Hello World!')
        yaml_with_key_mismatch = '\nconv2d_pw:\n  parameter_names_with_default_values:\n      NAME: conv2d_pw_1x1\n      TILE_SIZE_X: 1\n      TILE_SIZE_Y: 1\n  parameter_values:\n    - NAME: conv2d_pw_2x2\n      TILE_SIZE_X: 2\n      TILE_SIZE_Z: 2\n'
        generator = VulkanShaderGenerator()
        with tempfile.NamedTemporaryFile(mode='w') as fp:
            fp.write(yaml_with_key_mismatch)
            fp.flush()
            with self.assertRaisesRegex(KeyError, "Invalid keys {'TILE_SIZE_Z'}"):
                generator.add_params_yaml(fp.name)

    def test_missing_key_default_val(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        yaml_with_key_mismatch = '\nconv2d_pw:\n  parameter_names_with_default_values:\n      NAME: conv2d_pw_1x1\n      TILE_SIZE_X: 1\n      TILE_SIZE_Y: 1\n  parameter_values:\n    - NAME: conv2d_pw_1x2\n      TILE_SIZE_Y: 2\n'
        file_content = '\nx = $TILE_SIZE_X + $TILE_SIZE_Y\n'
        generator = VulkanShaderGenerator()
        with tempfile.NamedTemporaryFile(mode='w') as fp:
            fp.write(yaml_with_key_mismatch)
            fp.flush()
            generator.add_params_yaml(fp.name)
            with tempfile.TemporaryDirectory() as tmp_dir:
                template_file_name = os.path.join(tmp_dir, 'conv2d_pw.glslt')
                with open(template_file_name, 'w') as template_file:
                    template_file.write(file_content)
                    template_file.flush()
                    generator.generate(template_file.name, tmp_dir)
                    file_name_1 = os.path.join(tmp_dir, 'conv2d_pw_1x1.glsl')
                    file_name_2 = os.path.join(tmp_dir, 'conv2d_pw_1x2.glsl')
                    self.assertTrue(os.path.exists(file_name_1))
                    self.assertTrue(os.path.exists(file_name_2))
                    with open(file_name_1) as f:
                        contents = f.read()
                        self.assertTrue('1 + 1' in contents)
                    with open(file_name_2) as f:
                        contents = f.read()
                        self.assertTrue('1 + 2' in contents)