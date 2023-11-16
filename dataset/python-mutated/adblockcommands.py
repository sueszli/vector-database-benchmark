"""Commands relating to ad blocking."""
from qutebrowser.api import cmdutils
from qutebrowser.components import braveadblock, hostblock

@cmdutils.register()
def adblock_update() -> None:
    if False:
        return 10
    'Update block lists for both the host- and the Brave ad blocker.'
    if braveadblock.ad_blocker is not None:
        braveadblock.ad_blocker.adblock_update()
    hostblock.host_blocker.adblock_update()