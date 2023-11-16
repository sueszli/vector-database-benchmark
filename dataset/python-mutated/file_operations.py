"""Commands to perform operations on files"""
from __future__ import annotations
COMMAND_CATEGORY = 'file_operations'
COMMAND_CATEGORY_TITLE = 'File Operations'
import contextlib
import hashlib
import logging
import os
import os.path
from pathlib import Path
from typing import Iterator, Literal
from autogpt.agents.agent import Agent
from autogpt.agents.utils.exceptions import DuplicateOperationError
from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.memory.vector import MemoryItem, VectorMemory
from .decorators import sanitize_path_arg
from .file_context import open_file, open_folder
from .file_operations_utils import read_textual_file
logger = logging.getLogger(__name__)
Operation = Literal['write', 'append', 'delete']

def text_checksum(text: str) -> str:
    if False:
        i = 10
        return i + 15
    'Get the hex checksum for the given text.'
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def operations_from_log(log_path: str | Path) -> Iterator[tuple[Literal['write', 'append'], str, str] | tuple[Literal['delete'], str, None]]:
    if False:
        for i in range(10):
            print('nop')
    'Parse the file operations log and return a tuple containing the log entries'
    try:
        log = open(log_path, 'r', encoding='utf-8')
    except FileNotFoundError:
        return
    for line in log:
        line = line.replace('File Operation Logger', '').strip()
        if not line:
            continue
        (operation, tail) = line.split(': ', maxsplit=1)
        operation = operation.strip()
        if operation in ('write', 'append'):
            (path, checksum) = (x.strip() for x in tail.rsplit(' #', maxsplit=1))
            yield (operation, path, checksum)
        elif operation == 'delete':
            yield (operation, tail.strip(), None)
    log.close()

def file_operations_state(log_path: str | Path) -> dict[str, str]:
    if False:
        while True:
            i = 10
    'Iterates over the operations log and returns the expected state.\n\n    Parses a log file at file_manager.file_ops_log_path to construct a dictionary\n    that maps each file path written or appended to its checksum. Deleted files are\n    removed from the dictionary.\n\n    Returns:\n        A dictionary mapping file paths to their checksums.\n\n    Raises:\n        FileNotFoundError: If file_manager.file_ops_log_path is not found.\n        ValueError: If the log file content is not in the expected format.\n    '
    state = {}
    for (operation, path, checksum) in operations_from_log(log_path):
        if operation in ('write', 'append'):
            state[path] = checksum
        elif operation == 'delete':
            del state[path]
    return state

@sanitize_path_arg('file_path')
def is_duplicate_operation(operation: Operation, file_path: Path, agent: Agent, checksum: str | None=None) -> bool:
    if False:
        print('Hello World!')
    'Check if the operation has already been performed\n\n    Args:\n        operation: The operation to check for\n        file_path: The name of the file to check for\n        agent: The agent\n        checksum: The checksum of the contents to be written\n\n    Returns:\n        True if the operation has already been performed on the file\n    '
    with contextlib.suppress(ValueError):
        file_path = file_path.relative_to(agent.workspace.root)
    state = file_operations_state(agent.file_manager.file_ops_log_path)
    if operation == 'delete' and str(file_path) not in state:
        return True
    if operation == 'write' and state.get(str(file_path)) == checksum:
        return True
    return False

@sanitize_path_arg('file_path')
def log_operation(operation: Operation, file_path: Path, agent: Agent, checksum: str | None=None) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Log the file operation to the file_logger.log\n\n    Args:\n        operation: The operation to log\n        file_path: The name of the file the operation was performed on\n        checksum: The checksum of the contents to be written\n    '
    with contextlib.suppress(ValueError):
        file_path = file_path.relative_to(agent.workspace.root)
    log_entry = f'{operation}: {file_path}'
    if checksum is not None:
        log_entry += f' #{checksum}'
    logger.debug(f'Logging file operation: {log_entry}')
    append_to_file(agent.file_manager.file_ops_log_path, f'{log_entry}\n', agent, should_log=False)

@command('read_file', 'Read an existing file', {'filename': JSONSchema(type=JSONSchema.Type.STRING, description='The path of the file to read', required=True)})
@sanitize_path_arg('filename')
def read_file(filename: Path, agent: Agent) -> str:
    if False:
        while True:
            i = 10
    'Read a file and return the contents\n\n    Args:\n        filename (Path): The name of the file to read\n\n    Returns:\n        str: The contents of the file\n    '
    content = read_textual_file(filename, logger)
    return content

def ingest_file(filename: str, memory: VectorMemory) -> None:
    if False:
        print('Hello World!')
    '\n    Ingest a file by reading its content, splitting it into chunks with a specified\n    maximum length and overlap, and adding the chunks to the memory storage.\n\n    Args:\n        filename: The name of the file to ingest\n        memory: An object with an add() method to store the chunks in memory\n    '
    try:
        logger.info(f'Ingesting file {filename}')
        content = read_file(filename)
        file_memory = MemoryItem.from_text_file(content, filename)
        logger.debug(f'Created memory: {file_memory.dump(True)}')
        memory.add(file_memory)
        logger.info(f'Ingested {len(file_memory.e_chunks)} chunks from {filename}')
    except Exception as err:
        logger.warn(f"Error while ingesting file '{filename}': {err}")

@command('write_file', 'Write a file, creating it if necessary. If the file exists, it is overwritten.', {'filename': JSONSchema(type=JSONSchema.Type.STRING, description='The name of the file to write to', required=True), 'contents': JSONSchema(type=JSONSchema.Type.STRING, description='The contents to write to the file', required=True)}, aliases=['create_file'])
@sanitize_path_arg('filename')
async def write_to_file(filename: Path, contents: str, agent: Agent) -> str:
    """Write contents to a file

    Args:
        filename (Path): The name of the file to write to
        contents (str): The contents to write to the file

    Returns:
        str: A message indicating success or failure
    """
    checksum = text_checksum(contents)
    if is_duplicate_operation('write', filename, agent, checksum):
        raise DuplicateOperationError(f'File {filename.name} has already been updated.')
    directory = os.path.dirname(filename)
    os.makedirs(directory, exist_ok=True)
    await agent.workspace.write_file(filename, contents)
    log_operation('write', filename, agent, checksum)
    return f'File {filename.name} has been written successfully.'

def append_to_file(filename: Path, text: str, agent: Agent, should_log: bool=True) -> None:
    if False:
        i = 10
        return i + 15
    'Append text to a file\n\n    Args:\n        filename (Path): The name of the file to append to\n        text (str): The text to append to the file\n        should_log (bool): Should log output\n    '
    directory = os.path.dirname(filename)
    os.makedirs(directory, exist_ok=True)
    with open(filename, 'a') as f:
        f.write(text)
    if should_log:
        with open(filename, 'r') as f:
            checksum = text_checksum(f.read())
        log_operation('append', filename, agent, checksum=checksum)

@command('list_folder', 'List the items in a folder', {'folder': JSONSchema(type=JSONSchema.Type.STRING, description='The folder to list files in', required=True)})
@sanitize_path_arg('folder')
def list_folder(folder: Path, agent: Agent) -> list[str]:
    if False:
        print('Hello World!')
    'Lists files in a folder recursively\n\n    Args:\n        folder (Path): The folder to search in\n\n    Returns:\n        list[str]: A list of files found in the folder\n    '
    found_files = []
    for (root, _, files) in os.walk(folder):
        for file in files:
            if file.startswith('.'):
                continue
            relative_path = os.path.relpath(os.path.join(root, file), agent.workspace.root)
            found_files.append(relative_path)
    return found_files