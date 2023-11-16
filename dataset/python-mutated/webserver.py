"""Glances Web Interface (Bottle based)."""
from glances.globals import WINDOWS
from glances.processes import glances_processes
from glances.stats import GlancesStats
from glances.outputs.glances_bottle import GlancesBottle

class GlancesWebServer(object):
    """This class creates and manages the Glances Web server session."""

    def __init__(self, config=None, args=None):
        if False:
            i = 10
            return i + 15
        self.stats = GlancesStats(config, args)
        if not WINDOWS and args.no_kernel_threads:
            glances_processes.disable_kernel_threads()
        self.stats.update()
        self.web = GlancesBottle(config=config, args=args)

    def serve_forever(self):
        if False:
            print('Hello World!')
        'Main loop for the Web server.'
        self.web.start(self.stats)

    def end(self):
        if False:
            print('Hello World!')
        'End of the Web server.'
        self.web.end()
        self.stats.end()