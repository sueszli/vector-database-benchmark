"""
This script is responsible for cleaning the model section of the table of content by removing duplicates and sorting
the entries in alphabetical order.

Usage (from the root of the repo):

Check that the table of content is properly sorted (used in `make quality`):

```bash
python utils/check_doc_toc.py
```

Auto-sort the table of content if it is not properly sorted (used in `make style`):

```bash
python utils/check_doc_toc.py --fix_and_overwrite
```
"""
import argparse
from collections import defaultdict
from typing import List
import yaml
PATH_TO_TOC = 'docs/source/en/_toctree.yml'

def clean_model_doc_toc(model_doc: List[dict]) -> List[dict]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Cleans a section of the table of content of the model documentation (one specific modality) by removing duplicates\n    and sorting models alphabetically.\n\n    Args:\n        model_doc (`List[dict]`):\n            The list of dictionaries extracted from the `_toctree.yml` file for this specific modality.\n\n    Returns:\n        `List[dict]`: List of dictionaries like the input, but cleaned up and sorted.\n    '
    counts = defaultdict(int)
    for doc in model_doc:
        counts[doc['local']] += 1
    duplicates = [key for (key, value) in counts.items() if value > 1]
    new_doc = []
    for duplicate_key in duplicates:
        titles = list({doc['title'] for doc in model_doc if doc['local'] == duplicate_key})
        if len(titles) > 1:
            raise ValueError(f'{duplicate_key} is present several times in the documentation table of content at `docs/source/en/_toctree.yml` with different *Title* values. Choose one of those and remove the others.')
        new_doc.append({'local': duplicate_key, 'title': titles[0]})
    new_doc.extend([doc for doc in model_doc if counts[doc['local']] == 1])
    return sorted(new_doc, key=lambda s: s['title'].lower())

def check_model_doc(overwrite: bool=False):
    if False:
        print('Hello World!')
    '\n    Check that the content of the table of content in `_toctree.yml` is clean (no duplicates and sorted for the model\n    API doc) and potentially auto-cleans it.\n\n    Args:\n        overwrite (`bool`, *optional*, defaults to `False`):\n            Whether to just check if the TOC is clean or to auto-clean it (when `overwrite=True`).\n    '
    with open(PATH_TO_TOC, encoding='utf-8') as f:
        content = yaml.safe_load(f.read())
    api_idx = 0
    while content[api_idx]['title'] != 'API':
        api_idx += 1
    api_doc = content[api_idx]['sections']
    model_idx = 0
    while api_doc[model_idx]['title'] != 'Models':
        model_idx += 1
    model_doc = api_doc[model_idx]['sections']
    modalities_docs = [(idx, section) for (idx, section) in enumerate(model_doc) if 'sections' in section]
    diff = False
    for (idx, modality_doc) in modalities_docs:
        old_modality_doc = modality_doc['sections']
        new_modality_doc = clean_model_doc_toc(old_modality_doc)
        if old_modality_doc != new_modality_doc:
            diff = True
            if overwrite:
                model_doc[idx]['sections'] = new_modality_doc
    if diff:
        if overwrite:
            api_doc[model_idx]['sections'] = model_doc
            content[api_idx]['sections'] = api_doc
            with open(PATH_TO_TOC, 'w', encoding='utf-8') as f:
                f.write(yaml.dump(content, allow_unicode=True))
        else:
            raise ValueError('The model doc part of the table of content is not properly sorted, run `make style` to fix this.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fix_and_overwrite', action='store_true', help='Whether to fix inconsistencies.')
    args = parser.parse_args()
    check_model_doc(args.fix_and_overwrite)