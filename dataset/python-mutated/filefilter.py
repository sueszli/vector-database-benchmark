"""Filter imported files using a regular expression.
"""
import re
from beets import config
from beets.importer import SingletonImportTask
from beets.plugins import BeetsPlugin
from beets.util import bytestring_path

class FileFilterPlugin(BeetsPlugin):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.register_listener('import_task_created', self.import_task_created_event)
        self.config.add({'path': '.*'})
        self.path_album_regex = self.path_singleton_regex = re.compile(bytestring_path(self.config['path'].get()))
        if 'album_path' in self.config:
            self.path_album_regex = re.compile(bytestring_path(self.config['album_path'].get()))
        if 'singleton_path' in self.config:
            self.path_singleton_regex = re.compile(bytestring_path(self.config['singleton_path'].get()))

    def import_task_created_event(self, session, task):
        if False:
            return 10
        if task.items and len(task.items) > 0:
            items_to_import = []
            for item in task.items:
                if self.file_filter(item['path']):
                    items_to_import.append(item)
            if len(items_to_import) > 0:
                task.items = items_to_import
            else:
                return []
        elif isinstance(task, SingletonImportTask):
            if not self.file_filter(task.item['path']):
                return []
        return [task]

    def file_filter(self, full_path):
        if False:
            while True:
                i = 10
        'Checks if the configured regular expressions allow the import\n        of the file given in full_path.\n        '
        import_config = dict(config['import'])
        full_path = bytestring_path(full_path)
        if 'singletons' not in import_config or not import_config['singletons']:
            return self.path_album_regex.match(full_path) is not None
        else:
            return self.path_singleton_regex.match(full_path) is not None