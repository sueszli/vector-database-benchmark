from __future__ import annotations
from ..module_utils.my_util import question

def action_code():
    if False:
        i = 10
        return i + 15
    raise Exception('hello from my_action.py, this code should never execute')