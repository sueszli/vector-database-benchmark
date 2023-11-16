import iwlib
from libqtile.log_utils import logger
from libqtile.widget import base

def get_status(interface_name):
    if False:
        return 10
    interface = iwlib.get_iwconfig(interface_name)
    if 'stats' not in interface:
        return (None, None)
    quality = interface['stats']['quality']
    essid = bytes(interface['ESSID']).decode()
    return (essid, quality)

class Wlan(base.InLoopPollText):
    """
    Displays Wifi SSID and quality.

    Widget requirements: iwlib_.

    .. _iwlib: https://pypi.org/project/iwlib/
    """
    orientations = base.ORIENTATION_HORIZONTAL
    defaults = [('interface', 'wlan0', 'The interface to monitor'), ('update_interval', 1, 'The update interval.'), ('disconnected_message', 'Disconnected', 'String to show when the wlan is diconnected.'), ('format', '{essid} {quality}/70', 'Display format. For percents you can use "{essid} {percent:2.0%}"')]

    def __init__(self, **config):
        if False:
            while True:
                i = 10
        base.InLoopPollText.__init__(self, **config)
        self.add_defaults(Wlan.defaults)

    def poll(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            (essid, quality) = get_status(self.interface)
            disconnected = essid is None
            if disconnected:
                return self.disconnected_message
            return self.format.format(essid=essid, quality=quality, percent=quality / 70)
        except EnvironmentError:
            logger.error('%s: Probably your wlan device is switched off or  otherwise not present in your system.', self.__class__.__name__)