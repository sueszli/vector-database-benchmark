from st2common.runners.base_action import Action
from st2common.services.packs import get_pack_from_index

class ShowRemote(Action):
    """Get detailed information about an available pack from the StackStorm Exchange index"""

    def run(self, pack):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param pack: Pack Name to get info about\n        :type pack: ``str``\n        '
        return {'pack': get_pack_from_index(pack)}