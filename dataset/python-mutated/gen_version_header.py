import argparse
import os
from typing import cast, Dict, Tuple
Version = Tuple[int, int, int]

def parse_version(version: str) -> Version:
    if False:
        return 10
    '\n    Parses a version string into (major, minor, patch) version numbers.\n\n    Args:\n      version: Full version number string, possibly including revision / commit hash.\n\n    Returns:\n      An int 3-tuple of (major, minor, patch) version numbers.\n    '
    version_number_str = version
    for i in range(len(version)):
        c = version[i]
        if not (c.isdigit() or c == '.'):
            version_number_str = version[:i]
            break
    return cast(Version, tuple([int(n) for n in version_number_str.split('.')]))

def apply_replacements(replacements: Dict[str, str], text: str) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Applies the given replacements within the text.\n\n    Args:\n      replacements (dict): Mapping of str -> str replacements.\n      text (str): Text in which to make replacements.\n\n    Returns:\n      Text with replacements applied, if any.\n    '
    for (before, after) in replacements.items():
        text = text.replace(before, after)
    return text

def main(args: argparse.Namespace) -> None:
    if False:
        print('Hello World!')
    with open(args.version_path) as f:
        version = f.read().strip()
    (major, minor, patch) = parse_version(version)
    replacements = {'@TORCH_VERSION_MAJOR@': str(major), '@TORCH_VERSION_MINOR@': str(minor), '@TORCH_VERSION_PATCH@': str(patch)}
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.template_path) as input:
        with open(args.output_path, 'w') as output:
            for line in input.readlines():
                output.write(apply_replacements(replacements, line))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate version.h from version.h.in template')
    parser.add_argument('--template-path', required=True, help='Path to the template (i.e. version.h.in)')
    parser.add_argument('--version-path', required=True, help='Path to the file specifying the version')
    parser.add_argument('--output-path', required=True, help='Output path for expanded template (i.e. version.h)')
    args = parser.parse_args()
    main(args)