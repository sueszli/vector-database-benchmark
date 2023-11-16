import os
from hashlib import sha256
from typing import Any
from dagster import DataVersion, Definitions, InputContext, IOManager, OutputContext, asset, file_relative_path, observable_source_asset
from dagster._seven.temp_dir import get_system_temp_directory
from dagster._utils import mkdir_p

class NumberTextFileIOManager(IOManager):

    def __init__(self, root_dir: str):
        if False:
            print('Hello World!')
        self.root_dir = root_dir

    @staticmethod
    def with_directory(root_dir: str):
        if False:
            print('Hello World!')
        mkdir_p(root_dir)
        return NumberTextFileIOManager(root_dir=root_dir)

    def load_input(self, context: 'InputContext') -> int:
        if False:
            return 10
        asset_key_str = context.asset_key.to_user_string()
        full_path = os.path.join(self.root_dir, f'{asset_key_str}.txt')
        with open(full_path) as ff:
            return int(ff.read())

    def handle_output(self, context: 'OutputContext', obj: int) -> None:
        if False:
            i = 10
            return i + 15
        if context.op_def.name == 'input_number':
            return
        asset_key_str = context.asset_key.to_user_string()
        full_path = os.path.join(self.root_dir, f'{asset_key_str}.txt')
        with open(full_path, 'w') as ff:
            ff.write(str(obj))

def sha256_digest_from_str(string: str) -> str:
    if False:
        while True:
            i = 10
    hash_sig = sha256()
    hash_sig.update(bytearray(string, 'utf8'))
    return hash_sig.hexdigest()
FILE_PATH = file_relative_path(__file__, 'input_number.txt')

class ExternalFileInputManager(IOManager):

    def load_input(self, context: 'InputContext') -> object:
        if False:
            while True:
                i = 10
        with open(FILE_PATH) as ff:
            return int(ff.read())

    def handle_output(self, context: 'OutputContext', obj: Any) -> None:
        if False:
            return 10
        raise Exception('This should never be called')

@observable_source_asset(io_manager_key='external_file_input_manager')
def input_number():
    if False:
        print('Hello World!')
    with open(FILE_PATH) as ff:
        return DataVersion(sha256_digest_from_str(ff.read()))

@asset(code_version='v3')
def versioned_number(input_number):
    if False:
        print('Hello World!')
    return input_number

@asset(code_version='v1')
def multiplied_number(versioned_number):
    if False:
        for i in range(10):
            print('nop')
    return versioned_number * 2
defs = Definitions(assets=[input_number, versioned_number, multiplied_number], resources={'io_manager': NumberTextFileIOManager.with_directory(os.path.join(get_system_temp_directory(), 'versioning_example')), 'external_file_input_manager': ExternalFileInputManager()})