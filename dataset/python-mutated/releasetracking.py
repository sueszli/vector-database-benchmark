from sentry.plugins.base.v2 import Plugin2

class ReleaseTrackingPlugin(Plugin2):

    def get_plugin_type(self):
        if False:
            print('Hello World!')
        return 'release-tracking'

    def get_release_doc_html(self, hook_url):
        if False:
            while True:
                i = 10
        raise NotImplementedError