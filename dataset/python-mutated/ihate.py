"""Warns you about things you hate (or even blocks import)."""
from beets.importer import action
from beets.library import Album, Item, parse_query_string
from beets.plugins import BeetsPlugin
__author__ = 'baobab@heresiarch.info'
__version__ = '2.0'

def summary(task):
    if False:
        while True:
            i = 10
    'Given an ImportTask, produce a short string identifying the\n    object.\n    '
    if task.is_album:
        return f'{task.cur_artist} - {task.cur_album}'
    else:
        return f'{task.item.artist} - {task.item.title}'

class IHatePlugin(BeetsPlugin):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.register_listener('import_task_choice', self.import_task_choice_event)
        self.config.add({'warn': [], 'skip': []})

    @classmethod
    def do_i_hate_this(cls, task, action_patterns):
        if False:
            i = 10
            return i + 15
        'Process group of patterns (warn or skip) and returns True if\n        task is hated and not whitelisted.\n        '
        if action_patterns:
            for query_string in action_patterns:
                (query, _) = parse_query_string(query_string, Album if task.is_album else Item)
                if any((query.match(item) for item in task.imported_items())):
                    return True
        return False

    def import_task_choice_event(self, session, task):
        if False:
            for i in range(10):
                print('nop')
        skip_queries = self.config['skip'].as_str_seq()
        warn_queries = self.config['warn'].as_str_seq()
        if task.choice_flag == action.APPLY:
            if skip_queries or warn_queries:
                self._log.debug('processing your hate')
                if self.do_i_hate_this(task, skip_queries):
                    task.choice_flag = action.SKIP
                    self._log.info('skipped: {0}', summary(task))
                    return
                if self.do_i_hate_this(task, warn_queries):
                    self._log.info('you may hate this: {0}', summary(task))
            else:
                self._log.debug('nothing to do')
        else:
            self._log.debug('user made a decision, nothing to do')