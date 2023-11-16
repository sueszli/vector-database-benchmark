from pyload.core.datatypes.pyfile import PyFile
from ..base.addon import BaseAddon

class UnSkipOnFail(BaseAddon):
    __name__ = 'UnSkipOnFail'
    __type__ = 'addon'
    __version__ = '0.14'
    __status__ = 'testing'
    __config__ = [('enabled', 'bool', 'Activated', True)]
    __description__ = 'Restart skipped duplicates when download fails'
    __license__ = 'GPLv3'
    __authors__ = [('Walter Purcaro', 'vuolter@gmail.com')]

    def download_failed(self, pyfile):
        if False:
            while True:
                i = 10
        msg = self._('Looking for skipped duplicates of: {} (pid:{})')
        self.log_info(msg.format(pyfile.name, pyfile.package().id))
        link = self.find_duplicate(pyfile)
        if link:
            self.log_info(self._('Queue found duplicate: {} (pid:{})').format(link.name, link.package_id))
            pyfile_new = self._create_pyfile(link)
            pyfile_new.set_custom_status(self._('unskipped'), 'queued')
            self.pyload.files.save()
            pyfile_new.release()
        else:
            self.log_info(self._('No duplicates found'))

    def find_duplicate(self, pyfile):
        if False:
            for i in range(10):
                print('nop')
        '\n        Search all packages for duplicate links to "pyfile".\n\n        Duplicates are links that would overwrite "pyfile". To test on duplicity\n        the package-folder and link-name of twolinks are compared (link.name). So\n        this method returns a list of all links with equal package-folders and\n        filenames as "pyfile", but except the data for "pyfile" iotselöf. It does\n        MOT check the link\'s status.\n        '
        for pinfo in self.pyload.api.get_queue():
            if pinfo.folder != pyfile.package().folder:
                continue
            pdata = self.pyload.api.get_package_data(pinfo.pid)
            for link in pdata.links:
                if link.status != 4:
                    continue
                if link.name == pyfile.name and link.fid != pyfile.id:
                    return link

    def _create_pyfile(self, pylink):
        if False:
            print('Hello World!')
        return PyFile(self.pyload.files, pylink.fid, pylink.url, pylink.name, pylink.size, pylink.status, pylink.error, pylink.plugin, pylink.package_id, pylink.order)