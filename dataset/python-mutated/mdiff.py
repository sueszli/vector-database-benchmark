import logging
import os
import subprocess

class DiffError(Exception):
    pass

def diff_strings(output: str, expected_output: str) -> str:
    if False:
        i = 10
        return i + 15
    mdiff_path = 'web/tests/lib/mdiff.js'
    if not os.path.isfile(mdiff_path):
        msg = 'Cannot find mdiff for Markdown diff rendering'
        logging.error(msg)
        raise DiffError(msg)
    command = ['node', mdiff_path, output, expected_output]
    diff = subprocess.check_output(command, text=True)
    return diff