"""
Utility that checks whether the copies defined in the library match the original or not. This includes:
- All code commented with `# Copied from` comments,
- The list of models in the main README.md matches the ones in the localized READMEs,
- Files that are registered as full copies of one another in the `FULL_COPIES` constant of this script.

This also checks the list of models in the README is complete (has all models) and add a line to complete if there is
a model missing.

Use from the root of the repo with:

```bash
python utils/check_copies.py
```

for a check that will error in case of inconsistencies (used by `make repo-consistency`) or

```bash
python utils/check_copies.py --fix_and_overwrite
```

for a check that will fix all inconsistencies automatically (used by `make fix-copies`).
"""
import argparse
import glob
import os
import re
from typing import List, Optional, Tuple
import black
from doc_builder.style_doc import style_docstrings_in_code
from transformers.utils import direct_transformers_import
TRANSFORMERS_PATH = 'src/transformers'
MODEL_TEST_PATH = 'tests/models'
PATH_TO_DOCS = 'docs/source/en'
REPO_PATH = '.'
FULL_COPIES = {'examples/tensorflow/question-answering/utils_qa.py': 'examples/pytorch/question-answering/utils_qa.py', 'examples/flax/question-answering/utils_qa.py': 'examples/pytorch/question-answering/utils_qa.py'}
LOCALIZED_READMES = {'README.md': {'start_prompt': '🤗 Transformers currently provides the following architectures', 'end_prompt': '1. Want to contribute a new model?', 'format_model_list': '**[{title}]({model_link})** (from {paper_affiliations}) released with the paper {paper_title_link} by {paper_authors}.{supplements}'}, 'README_zh-hans.md': {'start_prompt': '🤗 Transformers 目前支持如下的架构', 'end_prompt': '1. 想要贡献新的模型？', 'format_model_list': '**[{title}]({model_link})** (来自 {paper_affiliations}) 伴随论文 {paper_title_link} 由 {paper_authors} 发布。{supplements}'}, 'README_zh-hant.md': {'start_prompt': '🤗 Transformers 目前支援以下的架構', 'end_prompt': '1. 想要貢獻新的模型？', 'format_model_list': '**[{title}]({model_link})** (from {paper_affiliations}) released with the paper {paper_title_link} by {paper_authors}.{supplements}'}, 'README_ko.md': {'start_prompt': '🤗 Transformers는 다음 모델들을 제공합니다', 'end_prompt': '1. 새로운 모델을 올리고 싶나요?', 'format_model_list': '**[{title}]({model_link})** ({paper_affiliations} 에서 제공)은 {paper_authors}.{supplements}의 {paper_title_link}논문과 함께 발표했습니다.'}, 'README_es.md': {'start_prompt': '🤗 Transformers actualmente proporciona las siguientes arquitecturas', 'end_prompt': '1. ¿Quieres aportar un nuevo modelo?', 'format_model_list': '**[{title}]({model_link})** (from {paper_affiliations}) released with the paper {paper_title_link} by {paper_authors}.{supplements}'}, 'README_ja.md': {'start_prompt': '🤗Transformersは現在、以下のアーキテクチャを提供しています', 'end_prompt': '1. 新しいモデルを投稿したいですか？', 'format_model_list': '**[{title}]({model_link})** ({paper_affiliations} から) {paper_authors}.{supplements} から公開された研究論文 {paper_title_link}'}, 'README_hd.md': {'start_prompt': '🤗 ट्रांसफॉर्मर वर्तमान में निम्नलिखित आर्किटेक्चर का समर्थन करते हैं', 'end_prompt': '1. एक नए मॉडल में योगदान देना चाहते हैं?', 'format_model_list': '**[{title}]({model_link})** ({paper_affiliations} से) {paper_authors}.{supplements} द्वाराअनुसंधान पत्र {paper_title_link} के साथ जारी किया गया'}}
transformers_module = direct_transformers_import(TRANSFORMERS_PATH)

def _should_continue(line: str, indent: str) -> bool:
    if False:
        print('Hello World!')
    return line.startswith(indent) or len(line.strip()) == 0 or re.search('^\\s*\\)(\\s*->.*:|:)\\s*$', line) is not None

def find_code_in_transformers(object_name: str, base_path: str=None) -> str:
    if False:
        while True:
            i = 10
    '\n    Find and return the source code of an object.\n\n    Args:\n        object_name (`str`):\n            The name of the object we want the source code of.\n        base_path (`str`, *optional*):\n            The path to the base folder where files are checked. If not set, it will be set to `TRANSFORMERS_PATH`.\n\n    Returns:\n        `str`: The source code of the object.\n    '
    parts = object_name.split('.')
    i = 0
    if base_path is None:
        base_path = TRANSFORMERS_PATH
    if base_path == MODEL_TEST_PATH:
        base_path = 'tests'
    module = parts[i]
    while i < len(parts) and (not os.path.isfile(os.path.join(base_path, f'{module}.py'))):
        i += 1
        if i < len(parts):
            module = os.path.join(module, parts[i])
    if i >= len(parts):
        raise ValueError(f'`object_name` should begin with the name of a module of transformers but got {object_name}.')
    with open(os.path.join(base_path, f'{module}.py'), 'r', encoding='utf-8', newline='\n') as f:
        lines = f.readlines()
    indent = ''
    line_index = 0
    for name in parts[i + 1:]:
        while line_index < len(lines) and re.search(f'^{indent}(class|def)\\s+{name}(\\(|\\:)', lines[line_index]) is None:
            line_index += 1
        indent += '    '
        line_index += 1
    if line_index >= len(lines):
        raise ValueError(f' {object_name} does not match any function or class in {module}.')
    start_index = line_index - 1
    while line_index < len(lines) and _should_continue(lines[line_index], indent):
        line_index += 1
    while len(lines[line_index - 1]) <= 1:
        line_index -= 1
    code_lines = lines[start_index:line_index]
    return ''.join(code_lines)
_re_copy_warning = re.compile('^(\\s*)#\\s*Copied from\\s+transformers\\.(\\S+\\.\\S+)\\s*($|\\S.*$)')
_re_copy_warning_for_test_file = re.compile('^(\\s*)#\\s*Copied from\\s+tests\\.(\\S+\\.\\S+)\\s*($|\\S.*$)')
_re_replace_pattern = re.compile('^\\s*(\\S+)->(\\S+)(\\s+.*|$)')
_re_fill_pattern = re.compile('<FILL\\s+[^>]*>')

def get_indent(code: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    Find the indent in the first non empty line in a code sample.\n\n    Args:\n        code (`str`): The code to inspect.\n\n    Returns:\n        `str`: The indent looked at (as string).\n    '
    lines = code.split('\n')
    idx = 0
    while idx < len(lines) and len(lines[idx]) == 0:
        idx += 1
    if idx < len(lines):
        return re.search('^(\\s*)\\S', lines[idx]).groups()[0]
    return ''

def blackify(code: str) -> str:
    if False:
        return 10
    '\n    Applies the black part of our `make style` command to some code.\n\n    Args:\n        code (`str`): The code to format.\n\n    Returns:\n        `str`: The formatted code.\n    '
    has_indent = len(get_indent(code)) > 0
    if has_indent:
        code = f'class Bla:\n{code}'
    mode = black.Mode(target_versions={black.TargetVersion.PY37}, line_length=119)
    result = black.format_str(code, mode=mode)
    (result, _) = style_docstrings_in_code(result)
    return result[len('class Bla:\n'):] if has_indent else result

def check_codes_match(observed_code: str, theoretical_code: str) -> Optional[int]:
    if False:
        print('Hello World!')
    '\n    Checks if two version of a code match with the exception of the class/function name.\n\n    Args:\n        observed_code (`str`): The code found.\n        theoretical_code (`str`): The code to match.\n\n    Returns:\n        `Optional[int]`: The index of the first line where there is a difference (if any) and `None` if the codes\n        match.\n    '
    observed_code_header = observed_code.split('\n')[0]
    theoretical_code_header = theoretical_code.split('\n')[0]
    _re_class_match = re.compile('class\\s+([^\\(:]+)(?:\\(|:)')
    _re_func_match = re.compile('def\\s+([^\\(]+)\\(')
    for re_pattern in [_re_class_match, _re_func_match]:
        if re_pattern.match(observed_code_header) is not None:
            observed_obj_name = re_pattern.search(observed_code_header).groups()[0]
            theoretical_name = re_pattern.search(theoretical_code_header).groups()[0]
            theoretical_code_header = theoretical_code_header.replace(theoretical_name, observed_obj_name)
    diff_index = 0
    if theoretical_code_header != observed_code_header:
        return 0
    diff_index = 1
    for (observed_line, theoretical_line) in zip(observed_code.split('\n')[1:], theoretical_code.split('\n')[1:]):
        if observed_line != theoretical_line:
            return diff_index
        diff_index += 1

def is_copy_consistent(filename: str, overwrite: bool=False) -> Optional[List[Tuple[str, int]]]:
    if False:
        return 10
    "\n    Check if the code commented as a copy in a file matches the original.\n\n    Args:\n        filename (`str`):\n            The name of the file to check.\n        overwrite (`bool`, *optional*, defaults to `False`):\n            Whether or not to overwrite the copies when they don't match.\n\n    Returns:\n        `Optional[List[Tuple[str, int]]]`: If `overwrite=False`, returns the list of differences as tuples `(str, int)`\n        with the name of the object having a diff and the line number where theere is the first diff.\n    "
    with open(filename, 'r', encoding='utf-8', newline='\n') as f:
        lines = f.readlines()
    diffs = []
    line_index = 0
    while line_index < len(lines):
        search_re = _re_copy_warning
        if filename.startswith('tests'):
            search_re = _re_copy_warning_for_test_file
        search = search_re.search(lines[line_index])
        if search is None:
            line_index += 1
            continue
        (indent, object_name, replace_pattern) = search.groups()
        base_path = TRANSFORMERS_PATH if not filename.startswith('tests') else MODEL_TEST_PATH
        theoretical_code = find_code_in_transformers(object_name, base_path=base_path)
        theoretical_indent = get_indent(theoretical_code)
        start_index = line_index + 1 if indent == theoretical_indent else line_index
        line_index = start_index + 1
        subcode = '\n'.join(theoretical_code.split('\n')[1:])
        indent = get_indent(subcode)
        should_continue = True
        while line_index < len(lines) and should_continue:
            line_index += 1
            if line_index >= len(lines):
                break
            line = lines[line_index]
            should_continue = _should_continue(line, indent) and re.search(f'^{indent}# End copy', line) is None
        while len(lines[line_index - 1]) <= 1:
            line_index -= 1
        observed_code_lines = lines[start_index:line_index]
        observed_code = ''.join(observed_code_lines)
        if len(replace_pattern) > 0:
            patterns = replace_pattern.replace('with', '').split(',')
            patterns = [_re_replace_pattern.search(p) for p in patterns]
            for pattern in patterns:
                if pattern is None:
                    continue
                (obj1, obj2, option) = pattern.groups()
                theoretical_code = re.sub(obj1, obj2, theoretical_code)
                if option.strip() == 'all-casing':
                    theoretical_code = re.sub(obj1.lower(), obj2.lower(), theoretical_code)
                    theoretical_code = re.sub(obj1.upper(), obj2.upper(), theoretical_code)
            theoretical_code = blackify(theoretical_code)
        diff_index = check_codes_match(observed_code, theoretical_code)
        if diff_index is not None:
            diffs.append([object_name, diff_index + start_index + 1])
            if overwrite:
                lines = lines[:start_index] + [theoretical_code] + lines[line_index:]
                line_index = start_index + 1
    if overwrite and len(diffs) > 0:
        print(f'Detected changes, rewriting {filename}.')
        with open(filename, 'w', encoding='utf-8', newline='\n') as f:
            f.writelines(lines)
    return diffs

def check_copies(overwrite: bool=False):
    if False:
        i = 10
        return i + 15
    "\n    Check every file is copy-consistent with the original. Also check the model list in the main README and other\n    READMEs are consistent.\n\n    Args:\n        overwrite (`bool`, *optional*, defaults to `False`):\n            Whether or not to overwrite the copies when they don't match.\n    "
    all_files = glob.glob(os.path.join(TRANSFORMERS_PATH, '**/*.py'), recursive=True)
    all_test_files = glob.glob(os.path.join(MODEL_TEST_PATH, '**/*.py'), recursive=True)
    all_files = list(all_files) + list(all_test_files)
    diffs = []
    for filename in all_files:
        new_diffs = is_copy_consistent(filename, overwrite)
        diffs += [f'- {filename}: copy does not match {d[0]} at line {d[1]}' for d in new_diffs]
    if not overwrite and len(diffs) > 0:
        diff = '\n'.join(diffs)
        raise Exception('Found the following copy inconsistencies:\n' + diff + '\nRun `make fix-copies` or `python utils/check_copies.py --fix_and_overwrite` to fix them.')
    check_model_list_copy(overwrite=overwrite)

def check_full_copies(overwrite: bool=False):
    if False:
        while True:
            i = 10
    "\n    Check the files that are full copies of others (as indicated in `FULL_COPIES`) are copy-consistent.\n\n    Args:\n        overwrite (`bool`, *optional*, defaults to `False`):\n            Whether or not to overwrite the copies when they don't match.\n    "
    diffs = []
    for (target, source) in FULL_COPIES.items():
        with open(source, 'r', encoding='utf-8') as f:
            source_code = f.read()
        with open(target, 'r', encoding='utf-8') as f:
            target_code = f.read()
        if source_code != target_code:
            if overwrite:
                with open(target, 'w', encoding='utf-8') as f:
                    print(f'Replacing the content of {target} by the one of {source}.')
                    f.write(source_code)
            else:
                diffs.append(f'- {target}: copy does not match {source}.')
    if not overwrite and len(diffs) > 0:
        diff = '\n'.join(diffs)
        raise Exception('Found the following copy inconsistencies:\n' + diff + '\nRun `make fix-copies` or `python utils/check_copies.py --fix_and_overwrite` to fix them.')

def get_model_list(filename: str, start_prompt: str, end_prompt: str) -> str:
    if False:
        while True:
            i = 10
    '\n    Extracts the model list from a README.\n\n    Args:\n        filename (`str`): The name of the README file to check.\n        start_prompt (`str`): The string to look for that introduces the model list.\n        end_prompt (`str`): The string to look for that ends the model list.\n\n    Returns:\n        `str`: The model list.\n    '
    with open(os.path.join(REPO_PATH, filename), 'r', encoding='utf-8', newline='\n') as f:
        lines = f.readlines()
    start_index = 0
    while not lines[start_index].startswith(start_prompt):
        start_index += 1
    start_index += 1
    result = []
    current_line = ''
    end_index = start_index
    while not lines[end_index].startswith(end_prompt):
        if lines[end_index].startswith('1.'):
            if len(current_line) > 1:
                result.append(current_line)
            current_line = lines[end_index]
        elif len(lines[end_index]) > 1:
            current_line = f'{current_line[:-1]} {lines[end_index].lstrip()}'
        end_index += 1
    if len(current_line) > 1:
        result.append(current_line)
    return ''.join(result)

def convert_to_localized_md(model_list: str, localized_model_list: str, format_str: str) -> Tuple[bool, str]:
    if False:
        while True:
            i = 10
    '\n    Compare the model list from the main README to the one in a localized README.\n\n    Args:\n        model_list (`str`): The model list in the main README.\n        localized_model_list (`str`): The model list in one of the localized README.\n        format_str (`str`):\n            The template for a model entry in the localized README (look at the `format_model_list` in the entries of\n            `LOCALIZED_READMES` for examples).\n\n    Returns:\n        `Tuple[bool, str]`: A tuple where the first value indicates if the READMEs match or not, and the second value\n        is the correct localized README.\n    '

    def _rep(match):
        if False:
            while True:
                i = 10
        (title, model_link, paper_affiliations, paper_title_link, paper_authors, supplements) = match.groups()
        return format_str.format(title=title, model_link=model_link, paper_affiliations=paper_affiliations, paper_title_link=paper_title_link, paper_authors=paper_authors, supplements=' ' + supplements.strip() if len(supplements) != 0 else '')
    _re_capture_meta = re.compile('\\*\\*\\[([^\\]]*)\\]\\(([^\\)]*)\\)\\*\\* \\(from ([^)]*)\\)[^\\[]*([^\\)]*\\)).*?by (.*?[A-Za-z\\*]{2,}?)\\. (.*)$')
    _re_capture_title_link = re.compile('\\*\\*\\[([^\\]]*)\\]\\(([^\\)]*)\\)\\*\\*')
    if len(localized_model_list) == 0:
        localized_model_index = {}
    else:
        try:
            localized_model_index = {re.search('\\*\\*\\[([^\\]]*)', line).groups()[0]: line for line in localized_model_list.strip().split('\n')}
        except AttributeError:
            raise AttributeError('A model name in localized READMEs cannot be recognized.')
    model_keys = [re.search('\\*\\*\\[([^\\]]*)', line).groups()[0] for line in model_list.strip().split('\n')]
    readmes_match = not any((k not in model_keys for k in localized_model_index))
    localized_model_index = {k: v for (k, v) in localized_model_index.items() if k in model_keys}
    for model in model_list.strip().split('\n'):
        (title, model_link) = _re_capture_title_link.search(model).groups()
        if title not in localized_model_index:
            readmes_match = False
            localized_model_index[title] = _re_capture_meta.sub(_rep, model + ' ')
        elif _re_fill_pattern.search(localized_model_index[title]) is not None:
            update = _re_capture_meta.sub(_rep, model + ' ')
            if update != localized_model_index[title]:
                readmes_match = False
                localized_model_index[title] = update
        else:
            localized_model_index[title] = _re_capture_title_link.sub(f'**[{title}]({model_link})**', localized_model_index[title], count=1)
    sorted_index = sorted(localized_model_index.items(), key=lambda x: x[0].lower())
    return (readmes_match, '\n'.join((x[1] for x in sorted_index)) + '\n')

def _find_text_in_file(filename: str, start_prompt: str, end_prompt: str) -> Tuple[str, int, int, List[str]]:
    if False:
        i = 10
        return i + 15
    '\n    Find the text in a file between two prompts.\n\n    Args:\n        filename (`str`): The name of the file to look into.\n        start_prompt (`str`): The string to look for that introduces the content looked for.\n        end_prompt (`str`): The string to look for that ends the content looked for.\n\n    Returns:\n        Tuple[str, int, int, List[str]]: The content between the two prompts, the index of the start line in the\n        original file, the index of the end line in the original file and the list of lines of that file.\n    '
    with open(filename, 'r', encoding='utf-8', newline='\n') as f:
        lines = f.readlines()
    start_index = 0
    while not lines[start_index].startswith(start_prompt):
        start_index += 1
    start_index += 1
    end_index = start_index
    while not lines[end_index].startswith(end_prompt):
        end_index += 1
    end_index -= 1
    while len(lines[start_index]) <= 1:
        start_index += 1
    while len(lines[end_index]) <= 1:
        end_index -= 1
    end_index += 1
    return (''.join(lines[start_index:end_index]), start_index, end_index, lines)

def check_model_list_copy(overwrite: bool=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Check the model lists in the README is consistent with the ones in the other READMES and also with `index.nmd`.\n\n    Args:\n        overwrite (`bool`, *optional*, defaults to `False`):\n            Whether or not to overwrite the copies when they don't match.\n    "
    with open(os.path.join(REPO_PATH, 'README.md'), 'r', encoding='utf-8', newline='\n') as f:
        readme = f.read()
    new_readme = readme.replace('https://huggingface.co/transformers', 'https://huggingface.co/docs/transformers')
    new_readme = new_readme.replace('https://huggingface.co/docs/main/transformers', 'https://huggingface.co/docs/transformers/main')
    if new_readme != readme:
        if overwrite:
            with open(os.path.join(REPO_PATH, 'README.md'), 'w', encoding='utf-8', newline='\n') as f:
                f.write(new_readme)
        else:
            raise ValueError('The main README contains wrong links to the documentation of Transformers. Run `make fix-copies` to automatically fix them.')
    md_list = get_model_list(filename='README.md', start_prompt=LOCALIZED_READMES['README.md']['start_prompt'], end_prompt=LOCALIZED_READMES['README.md']['end_prompt'])
    converted_md_lists = []
    for (filename, value) in LOCALIZED_READMES.items():
        _start_prompt = value['start_prompt']
        _end_prompt = value['end_prompt']
        _format_model_list = value['format_model_list']
        localized_md_list = get_model_list(filename, _start_prompt, _end_prompt)
        (readmes_match, converted_md_list) = convert_to_localized_md(md_list, localized_md_list, _format_model_list)
        converted_md_lists.append((filename, readmes_match, converted_md_list, _start_prompt, _end_prompt))
    for converted_md_list in converted_md_lists:
        (filename, readmes_match, converted_md, _start_prompt, _end_prompt) = converted_md_list
        if filename == 'README.md':
            continue
        if overwrite:
            (_, start_index, end_index, lines) = _find_text_in_file(filename=os.path.join(REPO_PATH, filename), start_prompt=_start_prompt, end_prompt=_end_prompt)
            with open(os.path.join(REPO_PATH, filename), 'w', encoding='utf-8', newline='\n') as f:
                f.writelines(lines[:start_index] + [converted_md] + lines[end_index:])
        elif not readmes_match:
            raise ValueError(f'The model list in the README changed and the list in `{filename}` has not been updated. Run `make fix-copies` to fix this.')
SPECIAL_MODEL_NAMES = {'Bert Generation': 'BERT For Sequence Generation', 'BigBird': 'BigBird-RoBERTa', 'Data2VecAudio': 'Data2Vec', 'Data2VecText': 'Data2Vec', 'Data2VecVision': 'Data2Vec', 'DonutSwin': 'Swin Transformer', 'Marian': 'MarianMT', 'MaskFormerSwin': 'Swin Transformer', 'OpenAI GPT-2': 'GPT-2', 'OpenAI GPT': 'GPT', 'Perceiver': 'Perceiver IO', 'SAM': 'Segment Anything', 'ViT': 'Vision Transformer (ViT)'}
MODELS_NOT_IN_README = ['BertJapanese', 'Encoder decoder', 'FairSeq Machine-Translation', 'HerBERT', 'RetriBERT', 'Speech Encoder decoder', 'Speech2Text', 'Speech2Text2', 'TimmBackbone', 'Vision Encoder decoder', 'VisionTextDualEncoder']
README_TEMPLATE = '1. **[{model_name}](https://huggingface.co/docs/main/transformers/model_doc/{model_type})** (from <FILL INSTITUTION>) released with the paper [<FILL PAPER TITLE>](<FILL ARKIV LINK>) by <FILL AUTHORS>.'

def check_readme(overwrite: bool=False):
    if False:
        i = 10
        return i + 15
    '\n    Check if the main README contains all the models in the library or not.\n\n    Args:\n        overwrite (`bool`, *optional*, defaults to `False`):\n            Whether or not to add an entry for the missing models using `README_TEMPLATE`.\n    '
    info = LOCALIZED_READMES['README.md']
    (models, start_index, end_index, lines) = _find_text_in_file(os.path.join(REPO_PATH, 'README.md'), info['start_prompt'], info['end_prompt'])
    models_in_readme = [re.search('\\*\\*\\[([^\\]]*)', line).groups()[0] for line in models.strip().split('\n')]
    model_names_mapping = transformers_module.models.auto.configuration_auto.MODEL_NAMES_MAPPING
    absents = [(key, name) for (key, name) in model_names_mapping.items() if SPECIAL_MODEL_NAMES.get(name, name) not in models_in_readme]
    absents = [(key, name) for (key, name) in absents if name not in MODELS_NOT_IN_README]
    if len(absents) > 0 and (not overwrite):
        print(absents)
        raise ValueError("The main README doesn't contain all models, run `make fix-copies` to fill it with the missing model(s) then complete the generated entries.\nIf the model is not supposed to be in the main README, add it to the list `MODELS_NOT_IN_README` in utils/check_copies.py.\nIf it has a different name in the repo than in the README, map the correspondence in `SPECIAL_MODEL_NAMES` in utils/check_copies.py.")
    new_models = [README_TEMPLATE.format(model_name=name, model_type=key) for (key, name) in absents]
    all_models = models.strip().split('\n') + new_models
    all_models = sorted(all_models, key=lambda x: re.search('\\*\\*\\[([^\\]]*)', x).groups()[0].lower())
    all_models = '\n'.join(all_models) + '\n'
    if all_models != models:
        if overwrite:
            print('Fixing the main README.')
            with open(os.path.join(REPO_PATH, 'README.md'), 'w', encoding='utf-8', newline='\n') as f:
                f.writelines(lines[:start_index] + [all_models] + lines[end_index:])
        else:
            raise ValueError('The main README model list is not properly sorted. Run `make fix-copies` to fix this.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fix_and_overwrite', action='store_true', help='Whether to fix inconsistencies.')
    args = parser.parse_args()
    check_readme(args.fix_and_overwrite)
    check_copies(args.fix_and_overwrite)
    check_full_copies(args.fix_and_overwrite)