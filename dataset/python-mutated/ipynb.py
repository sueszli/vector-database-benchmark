"""A module to support operations on ipynb files"""
import collections
import copy
import json
import re
import shutil
import tempfile
CodeLine = collections.namedtuple('CodeLine', ['cell_number', 'code'])

def is_python(cell):
    if False:
        for i in range(10):
            print('nop')
    'Checks if the cell consists of Python code.'
    return cell['cell_type'] == 'code' and cell['source'] and (not cell['source'][0].startswith('%%'))

def process_file(in_filename, out_filename, upgrader):
    if False:
        i = 10
        return i + 15
    'The function where we inject the support for ipynb upgrade.'
    print('Extracting code lines from original notebook')
    (raw_code, notebook) = _get_code(in_filename)
    raw_lines = [cl.code for cl in raw_code]
    with tempfile.NamedTemporaryFile('w', delete=False) as temp_file:
        (processed_file, new_file_content, log, process_errors) = upgrader.update_string_pasta('\n'.join(raw_lines), in_filename)
        if temp_file and processed_file:
            new_notebook = _update_notebook(notebook, raw_code, new_file_content.split('\n'))
            json.dump(new_notebook, temp_file)
        else:
            raise SyntaxError('Was not able to process the file: \n%s\n' % ''.join(log))
        files_processed = processed_file
        report_text = upgrader._format_log(log, in_filename, out_filename)
        errors = process_errors
    shutil.move(temp_file.name, out_filename)
    return (files_processed, report_text, errors)

def skip_magic(code_line, magic_list):
    if False:
        return 10
    'Checks if the cell has magic, that is not Python-based.\n\n  Args:\n      code_line: A line of Python code\n      magic_list: A list of jupyter "magic" exceptions\n\n  Returns:\n    If the line jupyter "magic" line, not Python line\n\n   >>> skip_magic(\'!ls -laF\', [\'%\', \'!\', \'?\'])\n  True\n  '
    for magic in magic_list:
        if code_line.startswith(magic):
            return True
    return False

def check_line_split(code_line):
    if False:
        for i in range(10):
            print('nop')
    'Checks if a line was split with `\\`.\n\n  Args:\n      code_line: A line of Python code\n\n  Returns:\n    If the line was split with `\\`\n\n  >>> skip_magic("!gcloud ml-engine models create ${MODEL} \\\\\\n")\n  True\n  '
    return re.search('\\\\\\s*\\n$', code_line)

def _get_code(input_file):
    if False:
        for i in range(10):
            print('nop')
    'Loads the ipynb file and returns a list of CodeLines.'
    raw_code = []
    with open(input_file) as in_file:
        notebook = json.load(in_file)
    cell_index = 0
    for cell in notebook['cells']:
        if is_python(cell):
            cell_lines = cell['source']
            is_line_split = False
            for (line_idx, code_line) in enumerate(cell_lines):
                if skip_magic(code_line, ['%', '!', '?']) or is_line_split:
                    code_line = '###!!!' + code_line
                    is_line_split = check_line_split(code_line)
                if is_line_split:
                    is_line_split = check_line_split(code_line)
                if line_idx == len(cell_lines) - 1 and code_line.endswith('\n'):
                    code_line = code_line.replace('\n', '###===')
                raw_code.append(CodeLine(cell_index, code_line.rstrip().replace('\n', '###===')))
            cell_index += 1
    return (raw_code, notebook)

def _update_notebook(original_notebook, original_raw_lines, updated_code_lines):
    if False:
        for i in range(10):
            print('nop')
    'Updates notebook, once migration is done.'
    new_notebook = copy.deepcopy(original_notebook)
    assert len(original_raw_lines) == len(updated_code_lines), 'The lengths of input and converted files are not the same: {} vs {}'.format(len(original_raw_lines), len(updated_code_lines))
    code_cell_idx = 0
    for cell in new_notebook['cells']:
        if not is_python(cell):
            continue
        applicable_lines = [idx for (idx, code_line) in enumerate(original_raw_lines) if code_line.cell_number == code_cell_idx]
        new_code = [updated_code_lines[idx] for idx in applicable_lines]
        cell['source'] = '\n'.join(new_code).replace('###!!!', '').replace('###===', '\n')
        code_cell_idx += 1
    return new_notebook