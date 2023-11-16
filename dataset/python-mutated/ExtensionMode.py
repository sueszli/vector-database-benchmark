import html
from ulauncher.api.result import Result
from ulauncher.api.shared.query import Query
from ulauncher.modes.BaseMode import BaseMode
from ulauncher.modes.extensions.DeferredResultRenderer import DeferredResultRenderer
from ulauncher.modes.extensions.ExtensionServer import ExtensionServer

class ExtensionMode(BaseMode):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.extensionServer = ExtensionServer.get_instance()
        self.deferredResultRenderer = DeferredResultRenderer.get_instance()

    def is_enabled(self, query: Query):
        if False:
            print('Hello World!')
        return bool(self.extensionServer.get_controller_by_keyword(query.keyword)) and ' ' in query

    def on_query_change(self, _query: Query):
        if False:
            i = 10
            return i + 15
        '\n        Triggered when user changes the query\n        '
        self.deferredResultRenderer.on_query_change()

    def handle_query(self, query: Query):
        if False:
            for i in range(10):
                print('nop')
        controller = self.extensionServer.get_controller_by_keyword(query.keyword)
        if not controller:
            msg = 'Invalid extension keyword'
            raise RuntimeError(msg)
        return controller.handle_query(query)

    def get_triggers(self):
        if False:
            print('Hello World!')
        '\n        :rtype: Iterable[:class:`~ulauncher.api.result.Result`]\n        '
        for controller in self.extensionServer.controllers.values():
            for (trigger_id, trigger) in controller.manifest.triggers.items():
                action = None
                if trigger.keyword is None:
                    action = {'type': 'event:launch_trigger', 'args': [trigger_id], 'ext_id': controller.extension_id}
                elif trigger.user_keyword:
                    action = f'{trigger.user_keyword} '
                if action:
                    yield Result(name=html.escape(trigger.name), description=html.escape(trigger.description), icon=controller.get_normalized_icon_path(trigger.icon), on_enter=action, searchable=True)