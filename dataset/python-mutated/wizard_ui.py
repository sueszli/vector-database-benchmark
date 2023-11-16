from typing import Dict
from typing import Union
from .cache import arg_cache
from .deps import Dependency
from .deps import check_grid_docker
from .deps import check_hagrid
from .deps import check_syft
from .deps import check_syft_deps
from .nb_output import NBOutput
steps = {}
steps['check_hagrid'] = False
steps['check_syft'] = False
steps['check_grid'] = False

def complete_install_wizard(output: Union[Dict[str, Dependency], NBOutput]) -> Union[Dict[str, Dependency], NBOutput]:
    if False:
        return 10
    flipped = arg_cache['install_wizard_complete']
    if not flipped:
        for (_, v) in steps.items():
            if v is False:
                return output
    arg_cache['install_wizard_complete'] = True
    if isinstance(output, NBOutput):
        if flipped != arg_cache['install_wizard_complete']:
            output.raw_output += '\n\n✅ You have completed the Install Wizard'
    return output

class WizardUI:

    @property
    def check_hagrid(self) -> Union[Dict[str, Dependency], NBOutput]:
        if False:
            print('Hello World!')
        steps['check_hagrid'] = True
        return complete_install_wizard(check_hagrid())

    @property
    def check_syft_deps(self) -> Union[Dict[str, Dependency], NBOutput]:
        if False:
            while True:
                i = 10
        steps['check_syft'] = True
        return complete_install_wizard(check_syft_deps())

    @property
    def check_syft(self) -> Union[Dict[str, Dependency], NBOutput]:
        if False:
            print('Hello World!')
        steps['check_syft'] = True
        return complete_install_wizard(check_syft())

    @property
    def check_syft_pre(self) -> Union[Dict[str, Dependency], NBOutput]:
        if False:
            for i in range(10):
                print('nop')
        steps['check_syft'] = True
        return complete_install_wizard(check_syft(pre=True))

    @property
    def check_grid_docker(self) -> Union[Dict[str, Dependency], NBOutput]:
        if False:
            while True:
                i = 10
        print('Deprecated. Please use .check_docker')
        return self.check_docker

    @property
    def check_docker(self) -> Union[Dict[str, Dependency], NBOutput]:
        if False:
            i = 10
            return i + 15
        steps['check_grid'] = True
        return complete_install_wizard(check_grid_docker())