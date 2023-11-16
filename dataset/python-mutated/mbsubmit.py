"""Aid in submitting information to MusicBrainz.

This plugin allows the user to print track information in a format that is
parseable by the MusicBrainz track parser [1]. Programmatic submitting is not
implemented by MusicBrainz yet.

[1] https://wiki.musicbrainz.org/History:How_To_Parse_Track_Listings
"""
from beets import ui
from beets.autotag import Recommendation
from beets.plugins import BeetsPlugin
from beets.ui.commands import PromptChoice
from beetsplug.info import print_data

class MBSubmitPlugin(BeetsPlugin):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.config.add({'format': '$track. $title - $artist ($length)', 'threshold': 'medium'})
        self.threshold = self.config['threshold'].as_choice({'none': Recommendation.none, 'low': Recommendation.low, 'medium': Recommendation.medium, 'strong': Recommendation.strong})
        self.register_listener('before_choose_candidate', self.before_choose_candidate_event)

    def before_choose_candidate_event(self, session, task):
        if False:
            return 10
        if task.rec <= self.threshold:
            return [PromptChoice('p', 'Print tracks', self.print_tracks)]

    def print_tracks(self, session, task):
        if False:
            print('Hello World!')
        for i in sorted(task.items, key=lambda i: i.track):
            print_data(None, i, self.config['format'].as_str())

    def commands(self):
        if False:
            while True:
                i = 10
        'Add beet UI commands for mbsubmit.'
        mbsubmit_cmd = ui.Subcommand('mbsubmit', help='Submit Tracks to MusicBrainz')

        def func(lib, opts, args):
            if False:
                while True:
                    i = 10
            items = lib.items(ui.decargs(args))
            self._mbsubmit(items)
        mbsubmit_cmd.func = func
        return [mbsubmit_cmd]

    def _mbsubmit(self, items):
        if False:
            print('Hello World!')
        'Print track information to be submitted to MusicBrainz.'
        for i in sorted(items, key=lambda i: i.track):
            print_data(None, i, self.config['format'].as_str())