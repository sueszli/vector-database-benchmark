"""Deprecation warning for the removed gmusic plugin."""
from beets.plugins import BeetsPlugin

class Gmusic(BeetsPlugin):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._log.warning("The 'gmusic' plugin has been removed following the shutdown of Google Play Music. Remove the plugin from your configuration to silence this warning.")