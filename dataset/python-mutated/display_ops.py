"""Contains UI methods for Nginx operations."""
import logging
from typing import Iterable
from typing import List
from typing import Optional
from certbot.display import util as display_util
from certbot_nginx._internal.obj import VirtualHost
logger = logging.getLogger(__name__)

def select_vhost_multiple(vhosts: Optional[Iterable[VirtualHost]]) -> List[VirtualHost]:
    if False:
        while True:
            i = 10
    'Select multiple Vhosts to install the certificate for\n    :param vhosts: Available Nginx VirtualHosts\n    :type vhosts: :class:`list` of type `~obj.Vhost`\n    :returns: List of VirtualHosts\n    :rtype: :class:`list`of type `~obj.Vhost`\n    '
    if not vhosts:
        return []
    tags_list = [vhost.display_repr() + '\n' for vhost in vhosts]
    if tags_list:
        tags_list[-1] = tags_list[-1][:-1]
    (code, names) = display_util.checklist('Which server blocks would you like to modify?', tags=tags_list, force_interactive=True)
    if code == display_util.OK:
        return_vhosts = _reversemap_vhosts(names, vhosts)
        return return_vhosts
    return []

def _reversemap_vhosts(names: Iterable[str], vhosts: Iterable[VirtualHost]) -> List[VirtualHost]:
    if False:
        i = 10
        return i + 15
    'Helper function for select_vhost_multiple for mapping string\n    representations back to actual vhost objects'
    return_vhosts = []
    for selection in names:
        for vhost in vhosts:
            if vhost.display_repr().strip() == selection.strip():
                return_vhosts.append(vhost)
    return return_vhosts