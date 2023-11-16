"""Define YAML spec."""
from conda.exceptions import EnvironmentFileEmpty, EnvironmentFileNotFound
from .. import env

class YamlFileSpec:
    _environment = None
    extensions = {'.yaml', '.yml'}

    def __init__(self, filename=None, **kwargs):
        if False:
            print('Hello World!')
        self.filename = filename
        self.msg = None

    def can_handle(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self._environment = env.from_file(self.filename)
            return True
        except EnvironmentFileNotFound as e:
            self.msg = str(e)
            return False
        except EnvironmentFileEmpty as e:
            self.msg = e.message
            return False
        except TypeError:
            self.msg = f'{self.filename} is not a valid yaml file.'
            return False

    @property
    def environment(self):
        if False:
            return 10
        if not self._environment:
            self.can_handle()
        return self._environment