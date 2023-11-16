import six
from st2common.runners.base_action import Action

class PacksTransformationAction(Action):

    def run(self, packs_status, packs_list=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param packs_status: Result from packs.download action.\n        :type: packs_status: ``dict``\n\n        :param packs_list: Names of the pack in Exchange, a git repo URL or local file system.\n        :type: packs_list: ``list``\n        '
        if not packs_list:
            packs_list = []
        packs = []
        for (pack_name, status) in six.iteritems(packs_status):
            if 'success' in status.lower():
                packs.append(pack_name)
        packs_list.extend(packs)
        return packs_list