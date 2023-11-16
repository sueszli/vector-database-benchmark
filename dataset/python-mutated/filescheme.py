"""Handler functions for file:... pages."""
import os
from qutebrowser.browser.webkit.network import networkreply
from qutebrowser.utils import jinja

def get_file_list(basedir, all_files, filterfunc):
    if False:
        return 10
    'Get a list of files filtered by a filter function and sorted by name.\n\n    Args:\n        basedir: The parent directory of all files.\n        all_files: The list of files to filter and sort.\n        filterfunc: The filter function.\n\n    Return:\n        A list of dicts. Each dict contains the name and absname keys.\n    '
    items = []
    for filename in all_files:
        absname = os.path.join(basedir, filename)
        if filterfunc(absname):
            items.append({'name': filename, 'absname': absname})
    return sorted(items, key=lambda v: v['name'].lower())

def is_root(directory):
    if False:
        while True:
            i = 10
    'Check if the directory is the root directory.\n\n    Args:\n        directory: The directory to check.\n\n    Return:\n        Whether the directory is a root directory or not.\n    '
    return os.path.dirname(directory) == directory

def parent_dir(directory):
    if False:
        for i in range(10):
            print('nop')
    'Return the parent directory for the given directory.\n\n    Args:\n        directory: The path to the directory.\n\n    Return:\n        The path to the parent directory.\n    '
    return os.path.normpath(os.path.join(directory, os.pardir))

def dirbrowser_html(path):
    if False:
        print('Hello World!')
    'Get the directory browser web page.\n\n    Args:\n        path: The directory path.\n\n    Return:\n        The HTML of the web page.\n    '
    title = 'Browse directory: {}'.format(path)
    if is_root(path):
        parent = None
    else:
        parent = parent_dir(path)
    try:
        all_files = os.listdir(path)
    except OSError as e:
        html = jinja.render('error.html', title='Error while reading directory', url='file:///{}'.format(path), error=str(e))
        return html.encode('UTF-8', errors='xmlcharrefreplace')
    files = get_file_list(path, all_files, os.path.isfile)
    directories = get_file_list(path, all_files, os.path.isdir)
    html = jinja.render('dirbrowser.html', title=title, url=path, parent=parent, files=files, directories=directories)
    return html.encode('UTF-8', errors='xmlcharrefreplace')

def handler(request, _operation, _current_url):
    if False:
        for i in range(10):
            print('nop')
    "Handler for a file:// URL.\n\n    Args:\n        request: QNetworkRequest to answer to.\n        _operation: The HTTP operation being done.\n        _current_url: The page we're on currently.\n\n    Return:\n        A QNetworkReply for directories, None for files.\n    "
    path = request.url().toLocalFile()
    try:
        if os.path.isdir(path):
            data = dirbrowser_html(path)
            return networkreply.FixedDataNetworkReply(request, data, 'text/html')
        return None
    except UnicodeEncodeError:
        return None