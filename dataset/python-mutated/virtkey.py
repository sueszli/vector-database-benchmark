"""
Accept a key from a hypervisor if the virt runner has already submitted an authorization request
"""
import logging
import salt.utils.virt
log = logging.getLogger(__name__)

def ext_pillar(hyper_id, pillar, name, key):
    if False:
        return 10
    '\n    Accept the key for the VM on the hyper, if authorized.\n    '
    vk = salt.utils.virt.VirtKey(hyper_id, name, __opts__)
    ok = vk.accept(key)
    pillar['virtkey'] = {name: ok}
    return {}