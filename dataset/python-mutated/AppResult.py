import operator
from os.path import basename
from gi.repository import Gio
from ulauncher.api.result import Result
from ulauncher.config import PATHS
from ulauncher.modes.apps.launch_app import launch_app
from ulauncher.utils.json_utils import json_load, json_save
app_starts_path = f'{PATHS.STATE}/app_starts.json'
app_starts = json_load(app_starts_path)

class AppResult(Result):
    """
    :param Gio.DesktopAppInfo app_info:
    """
    searchable = True

    def __init__(self, app_info):
        if False:
            while True:
                i = 10
        super().__init__(name=app_info.get_display_name(), icon=app_info.get_string('Icon') or '', description=app_info.get_description() or app_info.get_generic_name() or '', keywords=app_info.get_keywords(), _app_id=app_info.get_id(), _executable=basename(app_info.get_string('TryExec') or app_info.get_executable() or ''))

    @staticmethod
    def from_id(app_id):
        if False:
            print('Hello World!')
        try:
            app_info = Gio.DesktopAppInfo.new(app_id)
            return AppResult(app_info)
        except TypeError:
            return None

    @staticmethod
    def get_top_app_ids():
        if False:
            print('Hello World!')
        '\n        Returns list of app ids sorted by launch count\n        '
        sorted_tuples = sorted(app_starts.items(), key=operator.itemgetter(1), reverse=True)
        return [*map(operator.itemgetter(0), sorted_tuples)]

    @staticmethod
    def get_most_frequent(limit=5):
        if False:
            i = 10
            return i + 15
        '\n        Returns most frequent apps\n\n        TODO: rename to `get_most_recent` and update method to remove old apps\n\n        :param int limit: limit\n        :rtype: class:`ResultList`\n        '
        return list(filter(None, map(AppResult.from_id, AppResult.get_top_app_ids())))[:limit]

    def get_searchable_fields(self):
        if False:
            i = 10
            return i + 15
        frequency_weight = 1
        sorted_app_ids = AppResult.get_top_app_ids()
        count = len(sorted_app_ids)
        if count:
            index = sorted_app_ids.index(self._app_id) if self._app_id in sorted_app_ids else count
            frequency_weight = 1 - index / count * 0.1 + 0.05
        return [(self.name, 1 * frequency_weight), (self._executable, 0.8 * frequency_weight), (self.description, 0.7 * frequency_weight), *[(k, 0.6 * frequency_weight) for k in self.keywords]]

    def on_activation(self, *_):
        if False:
            print('Hello World!')
        starts = app_starts.get(self._app_id, 0)
        app_starts[self._app_id] = starts + 1
        json_save(app_starts, app_starts_path)
        return launch_app(self._app_id)