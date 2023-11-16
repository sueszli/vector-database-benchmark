from __future__ import print_function
import six
from st2common.content.utils import get_pack_base_path
from st2common.runners.base_action import Action
from st2common.util.pack import get_pack_metadata
from st2common.util.pack import get_pack_warnings

class GetPackWarnings(Action):

    def run(self, packs_status):
        if False:
            i = 10
            return i + 15
        '\n        :param packs_status: Name of the pack and download status.\n        :type: packs_status: ``dict``\n        '
        result = {}
        warning_list = []
        if not packs_status:
            return result
        for (pack, status) in six.iteritems(packs_status):
            if 'success' not in status.lower():
                continue
            warning = get_warnings(pack)
            if warning:
                warning_list.append(warning)
        result['warning_list'] = warning_list
        return result

def get_warnings(pack=None):
    if False:
        i = 10
        return i + 15
    result = None
    pack_path = get_pack_base_path(pack)
    try:
        pack_metadata = get_pack_metadata(pack_dir=pack_path)
        result = get_pack_warnings(pack_metadata)
    except Exception:
        print('Could not open pack.yaml at location %s' % pack_path)
    finally:
        return result