import re
import subprocess
from libqtile.widget import base

class CapsNumLockIndicator(base.ThreadPoolText):
    """Really simple widget to show the current Caps/Num Lock state."""
    defaults = [('update_interval', 0.5, 'Update Time in seconds.')]

    def __init__(self, **config):
        if False:
            while True:
                i = 10
        base.ThreadPoolText.__init__(self, '', **config)
        self.add_defaults(CapsNumLockIndicator.defaults)

    def get_indicators(self):
        if False:
            i = 10
            return i + 15
        'Return a list with the current state of the keys.'
        try:
            output = self.call_process(['xset', 'q'])
        except subprocess.CalledProcessError as err:
            output = err.output
            return []
        if output.startswith('Keyboard'):
            indicators = re.findall('(Caps|Num)\\s+Lock:\\s*(\\w*)', output)
            return indicators

    def poll(self):
        if False:
            for i in range(10):
                print('nop')
        'Poll content for the text box.'
        indicators = self.get_indicators()
        status = ' '.join([' '.join(indicator) for indicator in indicators])
        return status