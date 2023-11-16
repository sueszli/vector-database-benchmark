from hashlib import sha256
from dagster import DataVersion, Output, asset, file_relative_path, observable_source_asset

def sha256_digest_from_str(string: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    hash_sig = sha256()
    hash_sig.update(bytearray(string, 'utf8'))
    return hash_sig.hexdigest()
FILE_PATH = file_relative_path(__file__, 'input_number.txt')

@observable_source_asset
def input_number():
    if False:
        return 10
    with open(FILE_PATH) as ff:
        return DataVersion(sha256_digest_from_str(ff.read()))

@asset(code_version='v6', deps=[input_number])
def versioned_number():
    if False:
        i = 10
        return i + 15
    with open(FILE_PATH) as ff:
        value = int(ff.read())
        return Output(value, data_version=DataVersion(str(value)))

@asset(code_version='v1')
def multiplied_number(versioned_number):
    if False:
        while True:
            i = 10
    return versioned_number * 2