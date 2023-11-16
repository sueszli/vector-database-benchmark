from typing import List, Type
from orangewidget.utils.filedialogs import open_filename_dialog_save, open_filename_dialog, RecentPath, RecentPathsWidgetMixin, RecentPathsWComboMixin
from orangewidget.utils.filedialogs import fix_extension, format_filter, get_file_name, Compression
from Orange.data.io import FileFormat
from Orange.util import deprecated
__all__ = ['open_filename_dialog_save', 'open_filename_dialog', 'RecentPath', 'RecentPathsWidgetMixin', 'RecentPathsWComboMixin', 'stored_recent_paths_prepend']

@deprecated
def dialog_formats():
    if False:
        return 10
    '\n    Return readable file types for QFileDialogs.\n    '
    return 'All readable files ({});;'.format('*' + ' *'.join(FileFormat.readers.keys())) + ';;'.join(('{} (*{})'.format(f.DESCRIPTION, ' *'.join(f.EXTENSIONS)) for f in sorted(set(FileFormat.readers.values()), key=list(FileFormat.readers.values()).index)))

def stored_recent_paths_prepend(class_: Type[RecentPathsWidgetMixin], r: RecentPath) -> List[RecentPath]:
    if False:
        while True:
            i = 10
    '\n    Load existing stored defaults *recent_paths* and move or prepend\n    `r` to front.\n    '
    existing = get_stored_default_recent_paths(class_)
    if r in existing:
        existing.remove(r)
    return [r] + existing

def get_stored_default_recent_paths(class_: Type[RecentPathsWidgetMixin]):
    if False:
        for i in range(10):
            print('nop')
    recent_paths = []
    try:
        items = class_.settingsHandler.defaults.get('recent_paths', [])
        for item in items:
            if isinstance(item, RecentPath):
                recent_paths.append(item)
    except (AttributeError, KeyError, TypeError):
        pass
    return recent_paths