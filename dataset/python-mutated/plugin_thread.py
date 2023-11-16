import os
import pprint
import time
import traceback
from sys import exc_info
from threading import Thread
from types import MethodType

class PluginThread(Thread):
    """
    abstract base class for thread types.
    """

    def __init__(self, manager):
        if False:
            print('Hello World!')
        '\n        Constructor.\n        '
        super().__init__()
        self.active = False
        self.daemon = True
        self.pyload = manager.pyload
        self._ = manager._
        self.m = self.manager = manager

    def write_debug_report(self, pyfile):
        if False:
            i = 10
            return i + 15
        '\n        writes a debug report.\n\n        :return:``\n        '
        date = time.strftime('%Y-%m-%d_%H-%M-%S')
        dump_name = f'debug_{pyfile.pluginname}_{date}.zip'
        dump_filename = os.path.join(self.pyload.tempdir, dump_name)
        dump = self.get_debug_dump(pyfile)
        try:
            import zipfile
            with zipfile.ZipFile(dump_filename, 'w') as zip:
                try:
                    for entry in os.listdir(os.path.join(self.pyload.tempdir, pyfile.pluginname)):
                        try:
                            zip.write(os.path.join(self.pyload.tempdir, pyfile.pluginname, entry), os.path.join(pyfile.pluginname, entry))
                        except Exception:
                            pass
                except OSError:
                    pass
                info = zipfile.ZipInfo(os.path.join(pyfile.pluginname, 'debug_Report.txt'), time.gmtime())
                info.external_attr = 420 << 16
                zip.writestr(info, dump)
            if not os.stat(dump_filename).st_size:
                raise Exception('Empty Zipfile')
        except Exception as exc:
            self.pyload.log.debug(f'Error creating zip file: {exc}')
            dump_filename = dump_filename.replace('.zip', '.txt')
            with open(dump_filename, mode='w') as fp:
                fp.write(dump)
        self.pyload.log.info(self._('Debug Report written to {}').format(dump_filename))

    def get_debug_dump(self, pyfile):
        if False:
            return 10
        version = self.pyload.api.get_server_version()
        dump = f'pyLoad {version} Debug Report of {pyfile.pluginname} {pyfile.plugin.__version__} \n\nTRACEBACK:\n {traceback.format_exc()} \n\nFRAMESTACK:\n'
        tb = exc_info()[2]
        stack = []
        while tb:
            stack.append(tb.tb_frame)
            tb = tb.tb_next
        for frame in stack[1:]:
            dump += f'\n_frame {frame.f_code.co_name} in {frame.f_code.co_filename} at line {frame.f_lineno}\n'
            for (key, value) in frame.f_locals.items():
                dump += f'\t{key:20} = '
                try:
                    dump += pprint.pformat(value) + '\n'
                except Exception as exc:
                    dump += f'<ERROR WHILE PRINTING VALUE> {exc}\n'
            del frame
        del stack
        dump += '\n\n_PLUGIN OBJECT DUMP: \n\n'
        for name in dir(pyfile.plugin):
            attr = getattr(pyfile.plugin, name)
            if not name.endswith('__') and (not isinstance(attr, MethodType)):
                dump += f'\t{name:20} = '
                try:
                    dump += pprint.pformat(attr) + '\n'
                except Exception as exc:
                    dump += f'<ERROR WHILE PRINTING VALUE> {exc}\n'
        dump += '\n_PYFILE OBJECT DUMP: \n\n'
        for name in dir(pyfile):
            attr = getattr(pyfile, name)
            if not name.endswith('__') and (not isinstance(attr, MethodType)):
                dump += f'\t{name:20} = '
                try:
                    dump += pprint.pformat(attr) + '\n'
                except Exception as exc:
                    dump += f'<ERROR WHILE PRINTING VALUE> {exc}\n'
        if pyfile.pluginname in self.pyload.config.plugin:
            dump += '\n\nCONFIG: \n\n'
            dump += pprint.pformat(self.pyload.config.plugin[pyfile.pluginname]) + '\n'
        return dump

    def clean(self, pyfile):
        if False:
            i = 10
            return i + 15
        '\n        set thread as inactive and release pyfile.\n        '
        self.active = False
        pyfile.release()