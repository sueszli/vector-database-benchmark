import os
import uuid
import requests
from bottles.backend.logger import Logger
from bottles.backend.models.config import BottleConfig
from bottles.backend.utils.manager import ManagerUtils
logging = Logger()

class SteamGridDBManager:

    @staticmethod
    def get_game_grid(name: str, config: BottleConfig):
        if False:
            for i in range(10):
                print('nop')
        try:
            res = requests.get(f'https://steamgrid.usebottles.com/api/search/{name}')
        except:
            return
        if res.status_code == 200:
            return SteamGridDBManager.__save_grid(res.json(), config)

    @staticmethod
    def __save_grid(url: str, config: BottleConfig):
        if False:
            print('Hello World!')
        grids_path = os.path.join(ManagerUtils.get_bottle_path(config), 'grids')
        if not os.path.exists(grids_path):
            os.makedirs(grids_path)
        ext = url.split('.')[-1]
        filename = str(uuid.uuid4()) + '.' + ext
        path = os.path.join(grids_path, filename)
        try:
            r = requests.get(url)
            with open(path, 'wb') as f:
                f.write(r.content)
        except Exception:
            return
        return f'grid:{filename}'