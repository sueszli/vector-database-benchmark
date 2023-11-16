import inspect
import os
import re
from pathlib import Path
import torch
import torch._dynamo as torchdynamo
from torch._export.db.case import ExportCase, normalize_inputs
from torch._export.db.examples import all_examples
from torch.export import export
PWD = Path(__file__).absolute().parent
ROOT = Path(__file__).absolute().parent.parent.parent.parent
SOURCE = ROOT / Path('source')
EXPORTDB_SOURCE = SOURCE / Path('generated') / Path('exportdb')

def generate_example_rst(example_case: ExportCase):
    if False:
        while True:
            i = 10
    '\n    Generates the .rst files for all the examples in db/examples/\n    '
    model = example_case.model
    tags = ', '.join((f':doc:`{tag} <{tag}>`' for tag in example_case.tags))
    source_file = inspect.getfile(model.__class__) if isinstance(model, torch.nn.Module) else inspect.getfile(model)
    with open(source_file) as file:
        source_code = file.read()
    source_code = re.sub('from torch\\._export\\.db\\.case import .*\\n', '', source_code)
    source_code = re.sub('@export_case\\((.|\\n)*?\\)\\n', '', source_code)
    source_code = source_code.replace('\n', '\n    ')
    splitted_source_code = re.split('@export_rewrite_case.*\\n', source_code)
    assert len(splitted_source_code) in {1, 2}, f'more than one @export_rewrite_case decorator in {source_code}'
    title = f'{example_case.name}'
    doc_contents = f"{title}\n{'^' * len(title)}\n\n.. note::\n\n    Tags: {tags}\n\n    Support Level: {example_case.support_level.name}\n\nOriginal source code:\n\n.. code-block:: python\n\n    {splitted_source_code[0]}\n\nResult:\n\n.. code-block::\n\n"
    try:
        inputs = normalize_inputs(example_case.example_inputs)
        exported_program = export(model, inputs.args, inputs.kwargs, dynamic_shapes=example_case.dynamic_shapes)
        graph_output = str(exported_program)
        graph_output = re.sub('        # File(.|\\n)*?\\n', '', graph_output)
        graph_output = graph_output.replace('\n', '\n    ')
        output = f'    {graph_output}'
    except torchdynamo.exc.Unsupported as e:
        output = '    Unsupported: ' + str(e).split('\n')[0]
    except AssertionError as e:
        output = '    AssertionError: ' + str(e).split('\n')[0]
    except RuntimeError as e:
        output = '    RuntimeError: ' + str(e).split('\n')[0]
    doc_contents += output + '\n'
    if len(splitted_source_code) == 2:
        doc_contents += f'\n\nYou can rewrite the example above to something like the following:\n\n.. code-block:: python\n\n{splitted_source_code[1]}\n\n'
    return doc_contents

def generate_index_rst(example_cases, tag_to_modules, support_level_to_modules):
    if False:
        print('Hello World!')
    '\n    Generates the index.rst file\n    '
    support_contents = ''
    for (k, v) in support_level_to_modules.items():
        support_level = k.name.lower().replace('_', ' ').title()
        module_contents = '\n\n'.join(v)
        support_contents += f"\n{support_level}\n{'-' * len(support_level)}\n\n{module_contents}\n"
    tag_names = '\n    '.join((t for t in tag_to_modules.keys()))
    with open(os.path.join(PWD, 'blurb.txt')) as file:
        blurb = file.read()
    doc_contents = f'.. _torch.export_db:\n\nExportDB\n========\n\n{blurb}\n\n.. toctree::\n    :maxdepth: 1\n    :caption: Tags\n\n    {tag_names}\n\n{support_contents}\n'
    with open(os.path.join(EXPORTDB_SOURCE, 'index.rst'), 'w') as f:
        f.write(doc_contents)

def generate_tag_rst(tag_to_modules):
    if False:
        print('Hello World!')
    '\n    For each tag that shows up in each ExportCase.tag, generate an .rst file\n    containing all the examples that have that tag.\n    '
    for (tag, modules_rst) in tag_to_modules.items():
        doc_contents = f"{tag}\n{'=' * (len(tag) + 4)}\n"
        full_modules_rst = '\n\n'.join(modules_rst)
        full_modules_rst = re.sub('={3,}', lambda match: '-' * len(match.group()), full_modules_rst)
        doc_contents += full_modules_rst
        with open(os.path.join(EXPORTDB_SOURCE, f'{tag}.rst'), 'w') as f:
            f.write(doc_contents)

def generate_rst():
    if False:
        print('Hello World!')
    if not os.path.exists(EXPORTDB_SOURCE):
        os.makedirs(EXPORTDB_SOURCE)
    example_cases = all_examples()
    tag_to_modules = {}
    support_level_to_modules = {}
    for example_case in example_cases.values():
        doc_contents = generate_example_rst(example_case)
        for tag in example_case.tags:
            tag_to_modules.setdefault(tag, []).append(doc_contents)
        support_level_to_modules.setdefault(example_case.support_level, []).append(doc_contents)
    generate_tag_rst(tag_to_modules)
    generate_index_rst(example_cases, tag_to_modules, support_level_to_modules)
if __name__ == '__main__':
    generate_rst()