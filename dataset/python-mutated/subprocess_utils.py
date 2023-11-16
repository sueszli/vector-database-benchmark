from __future__ import annotations
import subprocess

def run_python_snippet(python_executable: str, code_to_run: str) -> str:
    if False:
        return 10
    "\n    Executes python code by calling python_executable with '-c' option.\n    "
    py_exec_cmd = (python_executable, '-c', code_to_run)
    return subprocess.check_output(py_exec_cmd, shell=False, text=True)