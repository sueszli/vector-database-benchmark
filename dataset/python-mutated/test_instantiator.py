import pathlib
import sys
from typing import Tuple
import torch
from torch import Tensor, nn
import torch.distributed as dist
if not dist.is_available():
    print('Distributed not available, skipping tests', file=sys.stderr)
    sys.exit(0)
from torch.distributed.nn.jit import instantiator
from torch.testing._internal.common_utils import TestCase, run_tests

@torch.jit.interface
class MyModuleInterface:

    def forward(self, tensor: Tensor, number: int, word: str='default') -> Tuple[Tensor, int, str]:
        if False:
            return 10
        pass

class MyModule(nn.Module):
    pass

def create_module():
    if False:
        print('Hello World!')
    return MyModule()

class TestInstantiator(TestCase):

    def test_get_arg_return_types_from_interface(self):
        if False:
            print('Hello World!')
        (args_str, arg_types_str, return_type_str) = instantiator.get_arg_return_types_from_interface(MyModuleInterface)
        self.assertEqual(args_str, 'tensor, number, word')
        self.assertEqual(arg_types_str, 'tensor: Tensor, number: int, word: str')
        self.assertEqual(return_type_str, 'Tuple[Tensor, int, str]')

    def test_instantiate_scripted_remote_module_template(self):
        if False:
            return 10
        dir_path = pathlib.Path(instantiator.INSTANTIATED_TEMPLATE_DIR_PATH)
        file_paths = dir_path.glob(f'{instantiator._FILE_PREFIX}*.py')
        for file_path in file_paths:
            file_path.unlink()
        file_paths = dir_path.glob(f'{instantiator._FILE_PREFIX}*.py')
        num_files_before = len(list(file_paths))
        self.assertEqual(num_files_before, 0)
        generated_module = instantiator.instantiate_scriptable_remote_module_template(MyModuleInterface)
        self.assertTrue(hasattr(generated_module, '_remote_forward'))
        self.assertTrue(hasattr(generated_module, '_generated_methods'))
        file_paths = dir_path.glob(f'{instantiator._FILE_PREFIX}*.py')
        num_files_after = len(list(file_paths))
        self.assertEqual(num_files_after, 1)

    def test_instantiate_non_scripted_remote_module_template(self):
        if False:
            for i in range(10):
                print('nop')
        dir_path = pathlib.Path(instantiator.INSTANTIATED_TEMPLATE_DIR_PATH)
        file_paths = dir_path.glob(f'{instantiator._FILE_PREFIX}*.py')
        for file_path in file_paths:
            file_path.unlink()
        file_paths = dir_path.glob(f'{instantiator._FILE_PREFIX}*.py')
        num_files_before = len(list(file_paths))
        self.assertEqual(num_files_before, 0)
        generated_module = instantiator.instantiate_non_scriptable_remote_module_template()
        self.assertTrue(hasattr(generated_module, '_remote_forward'))
        self.assertTrue(hasattr(generated_module, '_generated_methods'))
        file_paths = dir_path.glob(f'{instantiator._FILE_PREFIX}*.py')
        num_files_after = len(list(file_paths))
        self.assertEqual(num_files_after, 1)
if __name__ == '__main__':
    run_tests()