import os
import shlex
import shutil
import subprocess
from typing import Optional
from bottles.backend.logger import Logger
logging = Logger()

class CabExtract:
    """
    This class is used to extract a Windows cabinet file.
    It takes the cabinet file path and the destination name as input. Then it
    extracts the file in a new directory with the input name under the Bottles'
    temp directory.
    """
    requirements: bool = False
    path: str
    name: str
    files: list
    destination: str

    def __init__(self):
        if False:
            print('Hello World!')
        self.cabextract_bin = shutil.which('cabextract')

    def run(self, path: str, name: str='', files: Optional[list]=None, destination: str=''):
        if False:
            return 10
        if files is None:
            files = []
        self.path = path
        self.name = name
        self.files = files
        self.destination = shlex.quote(destination)
        self.name = self.name.replace('.', '_')
        if not self.__checks():
            return False
        return self.__extract()

    def __checks(self):
        if False:
            while True:
                i = 10
        if not os.path.exists(self.path) and '*' not in self.path:
            logging.error(f'Cab file {self.path} not found')
            return False
        return True

    def __extract(self) -> bool:
        if False:
            i = 10
            return i + 15
        if not os.path.exists(self.destination):
            os.makedirs(self.destination)
        try:
            if len(self.files) > 0:
                for file in self.files:
                    '\n                    if file already exists as a symlink, remove it\n                    preventing broken symlinks\n                    '
                    if os.path.exists(os.path.join(self.destination, file)):
                        if os.path.islink(os.path.join(self.destination, file)):
                            os.unlink(os.path.join(self.destination, file))
                    command = [self.cabextract_bin, f"-F '*{file}*'", f'-d {self.destination}', f'-q {self.path}']
                    command = ' '.join(command)
                    subprocess.Popen(command, shell=True).communicate()
                    if len(file.split('/')) > 1:
                        _file = file.split('/')[-1]
                        _dir = file.replace(_file, '')
                        if not os.path.exists(f'{self.destination}/{_file}'):
                            shutil.move(f'{self.destination}/{_dir}/{_file}', f'{self.destination}/{_file}')
            else:
                command_list = [self.cabextract_bin, f'-d {self.destination}', f'-q {self.path}']
                command = ' '.join(command_list)
                subprocess.Popen(command, shell=True).communicate()
            logging.info(f'Cabinet {self.name} extracted successfully')
            return True
        except Exception as exception:
            logging.error(f'Error while extracting cab file {self.path}:\n{exception}')
        return False