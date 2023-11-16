import os
from pathlib import PurePath
import traceback
from typing import List, Dict, Any
import fs
from fs.osfs import OSFS
from golem.rpc import utils as rpc_utils

class RemoteFS:

    def __init__(self, filesystem, upload_ctrl):
        if False:
            for i in range(10):
                print('nop')
        self._fs = filesystem
        self.upload_ctrl = upload_ctrl

    @rpc_utils.expose('fs.isdir')
    def fs_isdir(self, path) -> bool:
        if False:
            for i in range(10):
                print('nop')
        path = str(PurePath(path))
        return self._fs.getinfo(path).is_dir

    @rpc_utils.expose('fs.isfile')
    def fs_isfile(self, path) -> bool:
        if False:
            while True:
                i = 10
        path = str(PurePath(path))
        return self._fs.getinfo(path).is_file

    @rpc_utils.expose('fs.listdir')
    def fs_listdir(self, path) -> List[str]:
        if False:
            return 10
        try:
            return [str(PurePath(f)) for f in self._fs.listdir(path)]
        except (fs.errors.DirectoryExpected, fs.errors.ResourceNotFound):
            traceback.print_stack()
            return []

    @rpc_utils.expose('fs.mkdir')
    def fs_mkdir(self, path) -> None:
        if False:
            while True:
                i = 10
        path = str(PurePath(path))
        try:
            self._fs.makedir(path, recreate=True)
        except (fs.errors.DirectoryExpected, fs.errors.ResourceNotFound):
            traceback.print_stack()

    @rpc_utils.expose('fs.remove')
    def fs_remove(self, path) -> None:
        if False:
            return 10
        path = str(PurePath(path))
        return self._fs.remove(path)

    @rpc_utils.expose('fs.removetree')
    def fs_removetree(self, path) -> None:
        if False:
            return 10
        path = str(PurePath(path))
        try:
            return self._fs.removetree(path)
        except fs.errors.ResourceNotFound:
            pass

    @rpc_utils.expose('fs.meta')
    def fs_meta(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        return self.upload_ctrl.meta

    @rpc_utils.expose('fs.upload_id')
    def fs_upload_id(self, path) -> List[str]:
        if False:
            while True:
                i = 10
        path = str(PurePath(path))
        return self.upload_ctrl.open(path, 'wb')

    @rpc_utils.expose('fs.upload')
    def fs_upload(self, id_, data) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        return self.upload_ctrl.upload(id_, data)

    @rpc_utils.expose('fs.download_id')
    def fs_download_id(self, path) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        path = str(PurePath(path))
        return self.upload_ctrl.open(path, 'rb')

    @rpc_utils.expose('fs.download')
    def fs_download(self, id_) -> List[str]:
        if False:
            i = 10
            return i + 15
        return self.upload_ctrl.download(id_)

    def copy_files_to_tmp_location(self, files: List[str], dest: str) -> List[str]:
        if False:
            print('Hello World!')
        outs = []
        osfs = OSFS('/')
        self._fs.makedir(dest)
        for output in files:
            out_path = os.path.join(dest, os.path.basename(os.path.normpath(output)))
            if os.path.isfile(output):
                fs.copy.copy_file(osfs, output, self._fs, out_path)
            elif os.path.isdir(output):
                fs.copy.copy_dir(osfs, output, self._fs, out_path)
            else:
                pass
            outs.append(str(PurePath(out_path)))
        return outs