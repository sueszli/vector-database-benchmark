""" This cares about resolving module names at compile time compensating meta path based importers.

"""
from nuitka.__past__ import unicode
from nuitka.Options import isExperimental
from nuitka.PythonVersions import python_version
from nuitka.utils.ModuleNames import ModuleName
_six_moves = {'six.moves.builtins': '__builtin__' if python_version < 768 else 'builtins', 'six.moves.configparser': 'ConfigParser' if python_version < 768 else 'configparser', 'six.moves.copyreg': 'copy_reg' if python_version < 768 else 'copyreg', 'six.moves.dbm_gnu': 'gdbm' if python_version < 768 else 'dbm.gnu', 'six.moves._dummy_thread': 'dummy_thread' if python_version < 768 else '_dummy_thread', 'six.moves.http_cookiejar': 'cookielib' if python_version < 768 else 'http.cookiejar', 'six.moves.http_cookies': 'Cookie' if python_version < 768 else 'http.cookies', 'six.moves.html_entities': 'htmlentitydefs' if python_version < 768 else 'html.entities', 'six.moves.html_parser': 'HTMLParser' if python_version < 768 else 'html.parser', 'six.moves.http_client': 'httplib' if python_version < 768 else 'http.client', 'six.moves.email_mime_multipart': 'email.MIMEMultipart' if python_version < 768 else 'email.mime.multipart', 'six.moves.email_mime_nonmultipart': 'email.MIMENonMultipart' if python_version < 768 else 'email.mime.nonmultipart', 'six.moves.email_mime_text': 'email.MIMEText' if python_version < 768 else 'email.mime.text', 'six.moves.email_mime_base': 'email.MIMEBase' if python_version < 768 else 'email.mime.base', 'six.moves.BaseHTTPServer': 'BaseHTTPServer' if python_version < 768 else 'http.server', 'six.moves.CGIHTTPServer': 'CGIHTTPServer' if python_version < 768 else 'http.server', 'six.moves.SimpleHTTPServer': 'SimpleHTTPServer' if python_version < 768 else 'http.server', 'six.moves.cPickle': 'cPickle' if python_version < 768 else 'pickle', 'six.moves.queue': 'Queue' if python_version < 768 else 'queue', 'six.moves.reprlib': 'repr' if python_version < 768 else 'reprlib', 'six.moves.socketserver': 'SocketServer' if python_version < 768 else 'socketserver', 'six.moves._thread': 'thread' if python_version < 768 else '_thread', 'six.moves.tkinter': 'Tkinter' if python_version < 768 else 'tkinter', 'six.moves.tkinter_dialog': 'Dialog' if python_version < 768 else 'tkinter.dialog', 'six.moves.tkinter_filedialog': 'FileDialog' if python_version < 768 else 'tkinter.filedialog', 'six.moves.tkinter_scrolledtext': 'ScrolledText' if python_version < 768 else 'tkinter.scrolledtext', 'six.moves.tkinter_simpledialog': 'SimpleDialog' if python_version < 768 else 'tkinter.simpledialog', 'six.moves.tkinter_tix': 'Tix' if python_version < 768 else 'tkinter.tix', 'six.moves.tkinter_ttk': 'ttk' if python_version < 768 else 'tkinter.ttk', 'six.moves.tkinter_constants': 'Tkconstants' if python_version < 768 else 'tkinter.constants', 'six.moves.tkinter_dnd': 'Tkdnd' if python_version < 768 else 'tkinter.dnd', 'six.moves.tkinter_colorchooser': 'tkColorChooser' if python_version < 768 else 'tkinter_colorchooser', 'six.moves.tkinter_commondialog': 'tkCommonDialog' if python_version < 768 else 'tkinter_commondialog', 'six.moves.tkinter_tkfiledialog': 'tkFileDialog' if python_version < 768 else 'tkinter.filedialog', 'six.moves.tkinter_font': 'tkFont' if python_version < 768 else 'tkinter.font', 'six.moves.tkinter_messagebox': 'tkMessageBox' if python_version < 768 else 'tkinter.messagebox', 'six.moves.tkinter_tksimpledialog': 'tkSimpleDialog' if python_version < 768 else 'tkinter_tksimpledialog', 'six.moves.urllib_parse': None if python_version < 768 else 'urllib.parse', 'six.moves.urllib_error': None if python_version < 768 else 'urllib.error', 'six.moves.urllib_robotparser': 'robotparser' if python_version < 768 else 'urllib.robotparser', 'six.moves.xmlrpc_client': 'xmlrpclib' if python_version < 768 else 'xmlrpc.client', 'six.moves.xmlrpc_server': 'SimpleXMLRPCServer' if python_version < 768 else 'xmlrpc.server', 'six.moves.winreg': '_winreg' if python_version < 768 else 'winreg', 'six.moves.urllib.request': 'urllib2' if python_version < 768 else 'urllib.request'}

def resolveModuleName(module_name):
    if False:
        return 10
    'Resolve a module name to its real module name.'
    if str is not unicode and type(module_name) is unicode:
        module_name = str(module_name)
    module_name = ModuleName(module_name)
    if module_name.isBelowNamespace('bottle.ext'):
        return ModuleName('bottle_' + module_name.splitPackageName()[1].splitPackageName()[1].asString())
    elif module_name.isBelowNamespace('requests.packages'):
        return module_name.splitPackageName()[1].splitPackageName()[1]
    elif module_name.isBelowNamespace('pkg_resources.extern'):
        return ModuleName('pkg_resources._vendor').getChildNamed(module_name.getBasename())
    elif module_name in _six_moves:
        return ModuleName(_six_moves[module_name])
    elif module_name.hasNamespace('importlib_metadata') and python_version >= 896 and isExperimental('eliminate-backports'):
        return module_name.relocateModuleNamespace('importlib_metadata', 'importlib.metadata')
    elif module_name.hasNamespace('importlib_resources') and python_version >= 912 and isExperimental('eliminate-backports'):
        if module_name == 'importlib_resources.abc':
            return module_name
        return module_name.relocateModuleNamespace('importlib_resources', 'importlib.resources')
    else:
        return module_name