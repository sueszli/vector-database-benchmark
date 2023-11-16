"""Utilities initializing and managing Go modules."""
from __future__ import annotations
import os
from airflow.utils.process_utils import execute_in_subprocess

def init_module(go_module_name: str, go_module_path: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Initialize a Go module.\n\n    If a ``go.mod`` file already exists, this function will do nothing.\n\n    :param go_module_name: The name of the Go module to initialize.\n    :param go_module_path: The path to the directory containing the Go module.\n    '
    if os.path.isfile(os.path.join(go_module_path, 'go.mod')):
        return
    go_mod_init_cmd = ['go', 'mod', 'init', go_module_name]
    execute_in_subprocess(go_mod_init_cmd, cwd=go_module_path)

def install_dependencies(go_module_path: str) -> None:
    if False:
        i = 10
        return i + 15
    'Install dependencies for a Go module.\n\n    :param go_module_path: The path to the directory containing the Go module.\n    '
    go_mod_tidy = ['go', 'mod', 'tidy']
    execute_in_subprocess(go_mod_tidy, cwd=go_module_path)