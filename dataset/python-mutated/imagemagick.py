import os
import subprocess

class ImageMagickUtils:

    def __init__(self, path: str):
        if False:
            for i in range(10):
                print('nop')
        self.path = path

    @staticmethod
    def __validate_path(path: str):
        if False:
            while True:
                i = 10
        if os.path.exists(path):
            return False
        if os.path.isdir(path):
            return False
        return True

    def list_assets(self):
        if False:
            i = 10
            return i + 15
        cmd = f"identify '{self.path}'"
        try:
            res = subprocess.check_output(['bash', '-c', cmd])
        except:
            return []
        assets = []
        for r in res.decode().split('\n'):
            _r = r.replace(self.path, '').split()
            if len(_r) < 3:
                continue
            try:
                assets.append(int(_r[2].split('x')[0]))
            except ValueError:
                continue
        return assets

    def convert(self, dest: str, asset_size: int=256, resize: int=256, flatten: bool=True, alpha: bool=True, fallback: bool=True):
        if False:
            i = 10
            return i + 15
        if not self.__validate_path(dest):
            raise FileExistsError('Destination path already exists')
        assets = self.list_assets()
        asset_index = -1
        cmd = f"convert '{self.path}'"
        if asset_size not in assets:
            if not fallback:
                raise ValueError('Asset size not available')
            if len(assets) > 0:
                asset_size = max(assets)
                asset_index = assets.index(asset_size)
        else:
            asset_index = assets.index(asset_size)
        if asset_index != -1:
            cmd = f"convert '{self.path}[{asset_index}]'"
        if resize > 0:
            cmd += f' -thumbnail {resize}x{resize}'
        if alpha:
            cmd += ' -alpha on -background none'
        if flatten:
            cmd += ' -flatten'
        cmd += f" '{dest}'"
        subprocess.Popen(['bash', '-c', cmd])