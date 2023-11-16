"""pdf.js integration for qutebrowser."""
import os
from qutebrowser.qt.core import QUrl, QUrlQuery
from qutebrowser.utils import resources, javascript, jinja, standarddir, log, urlutils
from qutebrowser.config import config
_SYSTEM_PATHS = ['/usr/share/pdf.js/', '/app/share/pdf.js/', '/usr/share/javascript/pdf.js/', '/usr/share/javascript/pdf/']

class PDFJSNotFound(Exception):
    """Raised when no pdf.js installation is found.

    Attributes:
        path: path of the file that was requested but not found.
    """

    def __init__(self, path):
        if False:
            print('Hello World!')
        self.path = path
        message = "Path '{}' not found".format(path)
        super().__init__(message)

def generate_pdfjs_page(filename, url):
    if False:
        return 10
    'Return the html content of a page that displays a file with pdfjs.\n\n    Returns a string.\n\n    Args:\n        filename: The filename of the PDF to open.\n        url: The URL being opened.\n    '
    if not is_available():
        pdfjs_dir = os.path.join(standarddir.data(), 'pdfjs')
        return jinja.render('no_pdfjs.html', url=url.toDisplayString(), title='PDF.js not found', pdfjs_dir=pdfjs_dir)
    html = get_pdfjs_res('web/viewer.html').decode('utf-8')
    script = _generate_pdfjs_script(filename)
    html = html.replace('</body>', '</body><script>{}</script>'.format(script))
    pdfjs_script = '<script src="../build/pdf.js"></script>'
    html = html.replace(pdfjs_script, '<script>window.Response = undefined;</script>\n' + pdfjs_script)
    return html

def _generate_pdfjs_script(filename):
    if False:
        while True:
            i = 10
    'Generate the script that shows the pdf with pdf.js.\n\n    Args:\n        filename: The name of the file to open.\n    '
    url = QUrl('qute://pdfjs/file')
    url_query = QUrlQuery()
    url_query.addQueryItem('filename', filename)
    url.setQuery(url_query)
    js_url = javascript.to_js(url.toString(urlutils.FormatOption.ENCODED))
    return jinja.js_environment.from_string('\n        document.addEventListener("DOMContentLoaded", function() {\n            if (typeof window.PDFJS !== \'undefined\') {\n                // v1.x\n                window.PDFJS.verbosity = window.PDFJS.VERBOSITY_LEVELS.info;\n            } else {\n                // v2.x+\n                const options = window.PDFViewerApplicationOptions;\n                options.set(\'verbosity\', pdfjsLib.VerbosityLevel.INFOS);\n            }\n\n            if (typeof window.PDFView !== \'undefined\') {\n                // < v1.6\n                window.PDFView.open({{ url }});\n            } else {\n                // v1.6+\n                window.PDFViewerApplication.open({\n                    url: {{ url }},\n                    originalUrl: {{ url }}\n                });\n            }\n        });\n    ').render(url=js_url)

def get_pdfjs_res_and_path(path):
    if False:
        return 10
    'Get a pdf.js resource in binary format.\n\n    Returns a (content, path) tuple, where content is the file content and path\n    is the path where the file was found. If path is None, the bundled version\n    was used.\n\n    Args:\n        path: The path inside the pdfjs directory.\n    '
    path = path.lstrip('/')
    content = None
    file_path = None
    system_paths = _SYSTEM_PATHS + [os.path.join(standarddir.data(), 'pdfjs'), os.path.expanduser('~/.local/share/qutebrowser/pdfjs/')]
    names_to_try = [path, _remove_prefix(path)]
    for system_path in system_paths:
        (content, file_path) = _read_from_system(system_path, names_to_try)
        if content is not None:
            break
    if content is None:
        res_path = '3rdparty/pdfjs/{}'.format(path)
        try:
            content = resources.read_file_binary(res_path)
        except FileNotFoundError:
            raise PDFJSNotFound(path) from None
        except OSError as e:
            log.misc.warning('OSError while reading PDF.js file: {}'.format(e))
            raise PDFJSNotFound(path) from None
    return (content, file_path)

def get_pdfjs_res(path):
    if False:
        for i in range(10):
            print('nop')
    'Get a pdf.js resource in binary format.\n\n    Args:\n        path: The path inside the pdfjs directory.\n    '
    (content, _path) = get_pdfjs_res_and_path(path)
    return content

def _remove_prefix(path):
    if False:
        for i in range(10):
            print('nop')
    'Remove the web/ or build/ prefix of a pdfjs-file-path.\n\n    Args:\n        path: Path as string where the prefix should be stripped off.\n    '
    prefixes = {'web/', 'build/'}
    if any((path.startswith(prefix) for prefix in prefixes)):
        return path.split('/', maxsplit=1)[1]
    return path

def _read_from_system(system_path, names):
    if False:
        i = 10
        return i + 15
    'Try to read a file with one of the given names in system_path.\n\n    Returns a (content, path) tuple, where the path is the filepath that was\n    used.\n\n    Each file in names is considered equal, the first file that is found\n    is read and its binary content returned.\n\n    Returns (None, None) if no file could be found\n\n    Args:\n        system_path: The folder where the file should be searched.\n        names: List of possible file names.\n    '
    for name in names:
        try:
            full_path = os.path.join(system_path, name)
            with open(full_path, 'rb') as f:
                return (f.read(), full_path)
        except FileNotFoundError:
            continue
        except OSError as e:
            log.misc.warning('OSError while reading PDF.js file: {}'.format(e))
            continue
    return (None, None)

def is_available():
    if False:
        while True:
            i = 10
    'Return true if a pdfjs installation is available.'
    try:
        get_pdfjs_res('build/pdf.js')
        get_pdfjs_res('web/viewer.html')
    except PDFJSNotFound:
        return False
    else:
        return True

def should_use_pdfjs(mimetype, url):
    if False:
        while True:
            i = 10
    'Check whether PDF.js should be used.'
    is_download_url = url.scheme() == 'blob' and QUrl(url.path()).scheme() == 'qute'
    is_pdf = mimetype in ['application/pdf', 'application/x-pdf']
    config_enabled = config.instance.get('content.pdfjs', url=url)
    return is_pdf and (not is_download_url) and config_enabled

def get_main_url(filename: str, original_url: QUrl) -> QUrl:
    if False:
        print('Hello World!')
    'Get the URL to be opened to view a local PDF.'
    url = QUrl('qute://pdfjs/web/viewer.html')
    query = QUrlQuery()
    query.addQueryItem('filename', filename)
    query.addQueryItem('file', '')
    urlstr = original_url.toString(urlutils.FormatOption.ENCODED)
    query.addQueryItem('source', urlstr)
    url.setQuery(query)
    return url