"""
Utility that sorts the imports in the custom inits of Transformers. Transformers uses init files that delay the
import of an object to when it's actually needed. This is to avoid the main init importing all models, which would
make the line `import transformers` very slow when the user has all optional dependencies installed. The inits with
delayed imports have two halves: one definining a dictionary `_import_structure` which maps modules to the name of the
objects in each module, and one in `TYPE_CHECKING` which looks like a normal init for type-checkers. `isort` or `ruff`
properly sort the second half which looks like traditionl imports, the goal of this script is to sort the first half.

Use from the root of the repo with:

```bash
python utils/custom_init_isort.py
```

which will auto-sort the imports (used in `make style`).

For a check only (as used in `make quality`) run:

```bash
python utils/custom_init_isort.py --check_only
```
"""
import argparse
import os
import re
from typing import Any, Callable, List, Optional
PATH_TO_TRANSFORMERS = 'src/transformers'
_re_indent = re.compile('^(\\s*)\\S')
_re_direct_key = re.compile('^\\s*"([^"]+)":')
_re_indirect_key = re.compile('^\\s*_import_structure\\["([^"]+)"\\]')
_re_strip_line = re.compile('^\\s*"([^"]+)",\\s*$')
_re_bracket_content = re.compile('\\[([^\\]]+)\\]')

def get_indent(line: str) -> str:
    if False:
        print('Hello World!')
    'Returns the indent in  given line (as string).'
    search = _re_indent.search(line)
    return '' if search is None else search.groups()[0]

def split_code_in_indented_blocks(code: str, indent_level: str='', start_prompt: Optional[str]=None, end_prompt: Optional[str]=None) -> List[str]:
    if False:
        print('Hello World!')
    '\n    Split some code into its indented blocks, starting at a given level.\n\n    Args:\n        code (`str`): The code to split.\n        indent_level (`str`): The indent level (as string) to use for identifying the blocks to split.\n        start_prompt (`str`, *optional*): If provided, only starts splitting at the line where this text is.\n        end_prompt (`str`, *optional*): If provided, stops splitting at a line where this text is.\n\n    Warning:\n        The text before `start_prompt` or after `end_prompt` (if provided) is not ignored, just not split. The input `code`\n        can thus be retrieved by joining the result.\n\n    Returns:\n        `List[str]`: The list of blocks.\n    '
    index = 0
    lines = code.split('\n')
    if start_prompt is not None:
        while not lines[index].startswith(start_prompt):
            index += 1
        blocks = ['\n'.join(lines[:index])]
    else:
        blocks = []
    current_block = [lines[index]]
    index += 1
    while index < len(lines) and (end_prompt is None or not lines[index].startswith(end_prompt)):
        if len(lines[index]) > 0 and get_indent(lines[index]) == indent_level:
            if len(current_block) > 0 and get_indent(current_block[-1]).startswith(indent_level + ' '):
                current_block.append(lines[index])
                blocks.append('\n'.join(current_block))
                if index < len(lines) - 1:
                    current_block = [lines[index + 1]]
                    index += 1
                else:
                    current_block = []
            else:
                blocks.append('\n'.join(current_block))
                current_block = [lines[index]]
        else:
            current_block.append(lines[index])
        index += 1
    if len(current_block) > 0:
        blocks.append('\n'.join(current_block))
    if end_prompt is not None and index < len(lines):
        blocks.append('\n'.join(lines[index:]))
    return blocks

def ignore_underscore_and_lowercase(key: Callable[[Any], str]) -> Callable[[Any], str]:
    if False:
        while True:
            i = 10
    '\n    Wraps a key function (as used in a sort) to lowercase and ignore underscores.\n    '

    def _inner(x):
        if False:
            while True:
                i = 10
        return key(x).lower().replace('_', '')
    return _inner

def sort_objects(objects: List[Any], key: Optional[Callable[[Any], str]]=None) -> List[Any]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Sort a list of objects following the rules of isort (all uppercased first, camel-cased second and lower-cased\n    last).\n\n    Args:\n        objects (`List[Any]`):\n            The list of objects to sort.\n        key (`Callable[[Any], str]`, *optional*):\n            A function taking an object as input and returning a string, used to sort them by alphabetical order.\n            If not provided, will default to noop (so a `key` must be provided if the `objects` are not of type string).\n\n    Returns:\n        `List[Any]`: The sorted list with the same elements as in the inputs\n    '

    def noop(x):
        if False:
            while True:
                i = 10
        return x
    if key is None:
        key = noop
    constants = [obj for obj in objects if key(obj).isupper()]
    classes = [obj for obj in objects if key(obj)[0].isupper() and (not key(obj).isupper())]
    functions = [obj for obj in objects if not key(obj)[0].isupper()]
    key1 = ignore_underscore_and_lowercase(key)
    return sorted(constants, key=key1) + sorted(classes, key=key1) + sorted(functions, key=key1)

def sort_objects_in_import(import_statement: str) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Sorts the imports in a single import statement.\n\n    Args:\n        import_statement (`str`): The import statement in which to sort the imports.\n\n    Returns:\n        `str`: The same as the input, but with objects properly sorted.\n    '

    def _replace(match):
        if False:
            i = 10
            return i + 15
        imports = match.groups()[0]
        if ',' not in imports:
            return f'[{imports}]'
        keys = [part.strip().replace('"', '') for part in imports.split(',')]
        if len(keys[-1]) == 0:
            keys = keys[:-1]
        return '[' + ', '.join([f'"{k}"' for k in sort_objects(keys)]) + ']'
    lines = import_statement.split('\n')
    if len(lines) > 3:
        idx = 2 if lines[1].strip() == '[' else 1
        keys_to_sort = [(i, _re_strip_line.search(line).groups()[0]) for (i, line) in enumerate(lines[idx:-idx])]
        sorted_indices = sort_objects(keys_to_sort, key=lambda x: x[1])
        sorted_lines = [lines[x[0] + idx] for x in sorted_indices]
        return '\n'.join(lines[:idx] + sorted_lines + lines[-idx:])
    elif len(lines) == 3:
        if _re_bracket_content.search(lines[1]) is not None:
            lines[1] = _re_bracket_content.sub(_replace, lines[1])
        else:
            keys = [part.strip().replace('"', '') for part in lines[1].split(',')]
            if len(keys[-1]) == 0:
                keys = keys[:-1]
            lines[1] = get_indent(lines[1]) + ', '.join([f'"{k}"' for k in sort_objects(keys)])
        return '\n'.join(lines)
    else:
        import_statement = _re_bracket_content.sub(_replace, import_statement)
        return import_statement

def sort_imports(file: str, check_only: bool=True):
    if False:
        print('Hello World!')
    '\n    Sort the imports defined in the `_import_structure` of a given init.\n\n    Args:\n        file (`str`): The path to the init to check/fix.\n        check_only (`bool`, *optional*, defaults to `True`): Whether or not to just check (and not auto-fix) the init.\n    '
    with open(file, encoding='utf-8') as f:
        code = f.read()
    if '_import_structure' not in code:
        return
    main_blocks = split_code_in_indented_blocks(code, start_prompt='_import_structure = {', end_prompt='if TYPE_CHECKING:')
    for block_idx in range(1, len(main_blocks) - 1):
        block = main_blocks[block_idx]
        block_lines = block.split('\n')
        line_idx = 0
        while line_idx < len(block_lines) and '_import_structure' not in block_lines[line_idx]:
            if 'import dummy' in block_lines[line_idx]:
                line_idx = len(block_lines)
            else:
                line_idx += 1
        if line_idx >= len(block_lines):
            continue
        internal_block_code = '\n'.join(block_lines[line_idx:-1])
        indent = get_indent(block_lines[1])
        internal_blocks = split_code_in_indented_blocks(internal_block_code, indent_level=indent)
        pattern = _re_direct_key if '_import_structure = {' in block_lines[0] else _re_indirect_key
        keys = [pattern.search(b).groups()[0] if pattern.search(b) is not None else None for b in internal_blocks]
        keys_to_sort = [(i, key) for (i, key) in enumerate(keys) if key is not None]
        sorted_indices = [x[0] for x in sorted(keys_to_sort, key=lambda x: x[1])]
        count = 0
        reorderded_blocks = []
        for i in range(len(internal_blocks)):
            if keys[i] is None:
                reorderded_blocks.append(internal_blocks[i])
            else:
                block = sort_objects_in_import(internal_blocks[sorted_indices[count]])
                reorderded_blocks.append(block)
                count += 1
        main_blocks[block_idx] = '\n'.join(block_lines[:line_idx] + reorderded_blocks + [block_lines[-1]])
    if code != '\n'.join(main_blocks):
        if check_only:
            return True
        else:
            print(f'Overwriting {file}.')
            with open(file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(main_blocks))

def sort_imports_in_all_inits(check_only=True):
    if False:
        while True:
            i = 10
    '\n    Sort the imports defined in the `_import_structure` of all inits in the repo.\n\n    Args:\n        check_only (`bool`, *optional*, defaults to `True`): Whether or not to just check (and not auto-fix) the init.\n    '
    failures = []
    for (root, _, files) in os.walk(PATH_TO_TRANSFORMERS):
        if '__init__.py' in files:
            result = sort_imports(os.path.join(root, '__init__.py'), check_only=check_only)
            if result:
                failures = [os.path.join(root, '__init__.py')]
    if len(failures) > 0:
        raise ValueError(f'Would overwrite {len(failures)} files, run `make style`.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check_only', action='store_true', help='Whether to only check or fix style.')
    args = parser.parse_args()
    sort_imports_in_all_inits(check_only=args.check_only)