"""Creates freedesktop.org-compliant .directory files on an album level.
"""
from beets import ui
from beets.plugins import BeetsPlugin

class FreedesktopPlugin(BeetsPlugin):

    def commands(self):
        if False:
            return 10
        deprecated = ui.Subcommand('freedesktop', help='Print a message to redirect to thumbnails --dolphin')
        deprecated.func = self.deprecation_message
        return [deprecated]

    def deprecation_message(self, lib, opts, args):
        if False:
            i = 10
            return i + 15
        ui.print_("This plugin is deprecated. Its functionality is superseded by the 'thumbnails' plugin")
        ui.print_("'thumbnails --dolphin' replaces freedesktop. See doc & changelog for more information")