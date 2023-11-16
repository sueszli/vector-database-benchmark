import textwrap
from gpt_engineer.core.chat_to_files import to_files_and_memory, get_code_strings
from gpt_engineer.cli.file_selector import FILE_LIST_NAME
from unittest.mock import MagicMock

class DummyDBs:
    memory = {}
    logs = {}
    preprompts = {}
    input = {}
    workspace = {}
    archive = {}
    project_metadata = {}

def test_to_files_and_memory():
    if False:
        for i in range(10):
            print('nop')
    chat = textwrap.dedent('\n    This is a sample program.\n\n    file1.py\n    ```python\n    print("Hello, World!")\n    ```\n\n    file2.py\n    ```python\n    def add(a, b):\n        return a + b\n    ```\n    ')
    dbs = DummyDBs()
    to_files_and_memory(chat, dbs)
    assert dbs.memory['all_output.txt'] == chat
    expected_files = {'file1.py': 'print("Hello, World!")\n', 'file2.py': 'def add(a, b):\n    return a + b\n', 'README.md': '\nThis is a sample program.\n\nfile1.py\n'}
    for (file_name, file_content) in expected_files.items():
        assert dbs.workspace[file_name] == file_content

def test_to_files_with_square_brackets():
    if False:
        return 10
    chat = textwrap.dedent('\n    This is a sample program.\n\n    [file1.py]\n    ```python\n    print("Hello, World!")\n    ```\n\n    [file2.py]\n    ```python\n    def add(a, b):\n        return a + b\n    ```\n    ')
    dbs = DummyDBs()
    to_files_and_memory(chat, dbs)
    assert dbs.memory['all_output.txt'] == chat
    expected_files = {'file1.py': 'print("Hello, World!")\n', 'file2.py': 'def add(a, b):\n    return a + b\n', 'README.md': '\nThis is a sample program.\n\n[file1.py]\n'}
    for (file_name, file_content) in expected_files.items():
        assert dbs.workspace[file_name] == file_content

def test_files_with_brackets_in_name():
    if False:
        return 10
    chat = textwrap.dedent('\n    This is a sample program.\n\n    [id].jsx\n    ```javascript\n    console.log("Hello, World!")\n    ```\n    ')
    dbs = DummyDBs()
    to_files_and_memory(chat, dbs)
    assert dbs.memory['all_output.txt'] == chat
    expected_files = {'[id].jsx': 'console.log("Hello, World!")\n', 'README.md': '\nThis is a sample program.\n\n[id].jsx\n'}
    for (file_name, file_content) in expected_files.items():
        assert dbs.workspace[file_name] == file_content

def test_files_with_file_colon():
    if False:
        print('Hello World!')
    chat = textwrap.dedent('\n    This is a sample program.\n\n    [FILE: file1.py]\n    ```python\n    print("Hello, World!")\n    ```\n    ')
    dbs = DummyDBs()
    to_files_and_memory(chat, dbs)
    assert dbs.memory['all_output.txt'] == chat
    expected_files = {'file1.py': 'print("Hello, World!")\n', 'README.md': '\nThis is a sample program.\n\n[FILE: file1.py]\n'}
    for (file_name, file_content) in expected_files.items():
        assert dbs.workspace[file_name] == file_content

def test_files_with_back_tick():
    if False:
        print('Hello World!')
    chat = textwrap.dedent('\n    This is a sample program.\n\n    `file1.py`\n    ```python\n    print("Hello, World!")\n    ```\n    ')
    dbs = DummyDBs()
    to_files_and_memory(chat, dbs)
    assert dbs.memory['all_output.txt'] == chat
    expected_files = {'file1.py': 'print("Hello, World!")\n', 'README.md': '\nThis is a sample program.\n\n`file1.py`\n'}
    for (file_name, file_content) in expected_files.items():
        assert dbs.workspace[file_name] == file_content

def test_files_with_newline_between():
    if False:
        return 10
    chat = textwrap.dedent('\n    This is a sample program.\n\n    file1.py\n\n    ```python\n    print("Hello, World!")\n    ```\n    ')
    dbs = DummyDBs()
    to_files_and_memory(chat, dbs)
    assert dbs.memory['all_output.txt'] == chat
    expected_files = {'file1.py': 'print("Hello, World!")\n', 'README.md': '\nThis is a sample program.\n\nfile1.py\n\n'}
    for (file_name, file_content) in expected_files.items():
        assert dbs.workspace[file_name] == file_content

def test_files_with_newline_between_header():
    if False:
        for i in range(10):
            print('nop')
    chat = textwrap.dedent('\n    This is a sample program.\n\n    ## file1.py\n\n    ```python\n    print("Hello, World!")\n    ```\n    ')
    dbs = DummyDBs()
    to_files_and_memory(chat, dbs)
    assert dbs.memory['all_output.txt'] == chat
    expected_files = {'file1.py': 'print("Hello, World!")\n', 'README.md': '\nThis is a sample program.\n\n## file1.py\n\n'}
    for (file_name, file_content) in expected_files.items():
        assert dbs.workspace[file_name] == file_content

def test_get_code_strings(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    mock_db = MagicMock()
    mock_db.path = 'path/to'
    data = {'file1.txt': 'This is file 1 content', 'file2.txt': 'This is file 2 content'}
    mock_db.__getitem__ = lambda self, x: data.get(x)
    mock_db.__contains__ = lambda self, x: x in data
    mock_metadata_db = {FILE_LIST_NAME: 'path/to/file1.txt\npath/to/file2.txt'}

    def mock_get_all_files_in_dir(directory):
        if False:
            print('Hello World!')
        return ['path/to/file1.txt', 'path/to/file2.txt']

    def mock_open_file(path):
        if False:
            for i in range(10):
                print('nop')
        return f'File Data for file: {path}'
    monkeypatch.setattr('gpt_engineer.core.chat_to_files._get_all_files_in_dir', mock_get_all_files_in_dir)
    monkeypatch.setattr('gpt_engineer.core.chat_to_files._open_file', mock_open_file)
    result = get_code_strings(mock_db, mock_metadata_db)
    print(result)
    assert result['file1.txt'] == 'File Data for file: path/to/file1.txt'
    assert result['file2.txt'] == 'File Data for file: path/to/file2.txt'