from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Mapping
failed = False

def get_result_file_name(platform: str) -> Path:
    if False:
        while True:
            i = 10
    return Path(__file__).parent / f'verify_types_{platform.lower()}.json'

def run_pyright(platform: str) -> subprocess.CompletedProcess[bytes]:
    if False:
        return 10
    return subprocess.run(['pyright', f'--pythonplatform={platform}', '--pythonversion=3.8', '--verifytypes=trio', '--outputjson', '--ignoreexternal'], capture_output=True)

def check_less_than(key: str, current_dict: Mapping[str, int | float], last_dict: Mapping[str, int | float], /, invert: bool=False) -> None:
    if False:
        while True:
            i = 10
    global failed
    current = current_dict[key]
    last = last_dict[key]
    assert isinstance(current, (float, int))
    assert isinstance(last, (float, int))
    if current == last:
        return
    if (current > last) ^ invert:
        failed = True
        print('ERROR: ', end='')
    strcurrent = f'{current:.4}' if isinstance(current, float) else str(current)
    strlast = f'{last:.4}' if isinstance(last, float) else str(last)
    print(f"{key} has gone {('down' if current < last else 'up')} from {strlast} to {strcurrent}")

def check_zero(key: str, current_dict: Mapping[str, float]) -> None:
    if False:
        for i in range(10):
            print('nop')
    global failed
    if current_dict[key] != 0:
        failed = True
        print(f'ERROR: {key} is {current_dict[key]}')

def check_type(args: argparse.Namespace, platform: str) -> int:
    if False:
        for i in range(10):
            print('nop')
    print('*' * 20, "\nChecking type completeness hasn't gone down...")
    res = run_pyright(platform)
    current_result = json.loads(res.stdout)
    py_typed_file: Path | None = None
    if current_result['generalDiagnostics'] and current_result['generalDiagnostics'][0]['message'] == 'No py.typed file found':
        print('creating py.typed')
        py_typed_file = Path(current_result['typeCompleteness']['packageRootDirectory']) / 'py.typed'
        py_typed_file.write_text('')
        res = run_pyright(platform)
        current_result = json.loads(res.stdout)
    if res.stderr:
        print(res.stderr)
    last_result = json.loads(get_result_file_name(platform).read_text())
    for key in ('errorCount', 'warningCount', 'informationCount'):
        check_zero(key, current_result['summary'])
    for (key, invert) in (('missingFunctionDocStringCount', False), ('missingClassDocStringCount', False), ('missingDefaultParamCount', False), ('completenessScore', True)):
        check_less_than(key, current_result['typeCompleteness'], last_result['typeCompleteness'], invert=invert)
    for (key, invert) in (('withUnknownType', False), ('withAmbiguousType', False), ('withKnownType', True)):
        check_less_than(key, current_result['typeCompleteness']['exportedSymbolCounts'], last_result['typeCompleteness']['exportedSymbolCounts'], invert=invert)
    if args.overwrite_file:
        print('Overwriting file')
        del current_result['time']
        del current_result['summary']['timeInSec']
        del current_result['version']
        for key in ('moduleRootDirectory', 'packageRootDirectory', 'pyTypedPath'):
            del current_result['typeCompleteness'][key]
        new_symbols: list[dict[str, str]] = []
        for symbol in current_result['typeCompleteness']['symbols']:
            if symbol['diagnostics']:
                new_symbols.extend(({'name': symbol['name'], 'message': diagnostic['message']} for diagnostic in symbol['diagnostics']))
                continue
        new_symbols.sort(key=lambda module: module.get('name', ''))
        current_result['generalDiagnostics'].sort()
        current_result['typeCompleteness']['modules'].sort(key=lambda module: module.get('name', ''))
        del current_result['typeCompleteness']['symbols']
        current_result['typeCompleteness']['diagnostics'] = new_symbols
        with open(get_result_file_name(platform), 'w') as file:
            json.dump(current_result, file, sort_keys=True, indent=2)
            file.write('\n')
    if py_typed_file is not None:
        print('deleting py.typed')
        py_typed_file.unlink()
    print('*' * 20)
    return int(failed)

def main(args: argparse.Namespace) -> int:
    if False:
        i = 10
        return i + 15
    res = 0
    for platform in ('Linux', 'Windows', 'Darwin'):
        res += check_type(args, platform)
    return res
parser = argparse.ArgumentParser()
parser.add_argument('--overwrite-file', action='store_true', default=False)
parser.add_argument('--full-diagnostics-file', type=Path, default=None)
args = parser.parse_args()
assert __name__ == '__main__', 'This script should be run standalone'
sys.exit(main(args))