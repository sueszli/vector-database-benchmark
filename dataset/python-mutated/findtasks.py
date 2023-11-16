"""
Source code analysis utilities.
"""
import re
from spyder.config.base import get_debug_level
DEBUG_EDITOR = get_debug_level() >= 3
TASKS_PATTERN = '(^|#)[ ]*(TODO|FIXME|XXX|HINT|TIP|@todo|HACK|BUG|OPTIMIZE|!!!|\\?\\?\\?)([^#]*)'

def find_tasks(source_code):
    if False:
        return 10
    'Find tasks in source code (TODO, FIXME, XXX, ...).'
    results = []
    for (line, text) in enumerate(source_code.splitlines()):
        for todo in re.findall(TASKS_PATTERN, text):
            todo_text = todo[-1].strip(' :').capitalize() if todo[-1] else todo[-2]
            results.append((todo_text, line + 1))
    return results