import logging
from voltron.view import *
from voltron.plugin import *
from voltron.api import *
log = logging.getLogger('view')

class BacktraceView(TerminalView):

    def build_requests(self):
        if False:
            i = 10
            return i + 15
        return [api_request('command', block=self.block, command='bt')]

    def render(self, results):
        if False:
            for i in range(10):
                print('nop')
        [res] = results
        self.title = '[backtrace]'
        if res.timed_out:
            return
        if res and res.is_success:
            self.body = res.output
        else:
            log.error('Error getting backtrace: {}'.format(res.message))
            self.body = self.colour(res.message, 'red')
        super(BacktraceView, self).render(results)

class BacktraceViewPlugin(ViewPlugin):
    plugin_type = 'view'
    name = 'backtrace'
    aliases = ('t', 'bt', 'back')
    view_class = BacktraceView