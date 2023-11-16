import lief
import subprocess
from utils import get_sample, is_apple_m1, sign, chmod_exe

def process(target: lief.MachO.Binary):
    if False:
        print('Hello World!')
    assert target.has(lief.MachO.LOAD_COMMAND_TYPES.DYLD_EXPORTS_TRIE)
    exports = target.get(lief.MachO.LOAD_COMMAND_TYPES.DYLD_EXPORTS_TRIE)
    assert exports.data_offset == 459384
    entries = list(exports.exports)
    entries = sorted(entries, key=lambda e: e.symbol.name)
    assert len(entries) == 885
    assert entries[1].symbol.name == '_main'
    assert entries[1].address == 17744
    assert entries[843].symbol.name == '_psa_its_remove'
    assert entries[843].address == 252620

def test_basic():
    if False:
        return 10
    fat = lief.MachO.parse(get_sample('MachO/9edfb04c55289c6c682a25211a4b30b927a86fe50b014610d04d6055bd4ac23d_crypt_and_hash.macho'))
    target = fat.take(lief.MachO.CPU_TYPES.ARM64)
    process(target)
    assert target.get(lief.MachO.LOAD_COMMAND_TYPES.DYLD_EXPORTS_TRIE).data_size == 16728

def test_write(tmp_path):
    if False:
        return 10
    binary_name = 'crypt_and_hash'
    fat = lief.MachO.parse(get_sample('MachO/9edfb04c55289c6c682a25211a4b30b927a86fe50b014610d04d6055bd4ac23d_crypt_and_hash.macho'))
    target = fat.take(lief.MachO.CPU_TYPES.ARM64)
    output = f'{tmp_path}/{binary_name}.built'
    target.write(output)
    target = lief.parse(output)
    process(target)
    (valid, err) = lief.MachO.check_layout(target)
    assert valid, err
    if is_apple_m1():
        chmod_exe(output)
        sign(output)
        with subprocess.Popen([output], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
            stdout = proc.stdout.read()
            assert 'CAMELLIA-256-CCM*-NO-TAG' in stdout
            assert 'AES-128-CCM*-NO-TAG' in stdout