import logging
import time
import uuid
from dataclasses import dataclass, field
from PyQt5 import uic
from tribler.core.components.metadata_store.db.serialization import REGULAR_TORRENT
from tribler.core.utilities.utilities import Query, to_fts_query
from tribler.gui.network.request_manager import request_manager
from tribler.gui.sentry_mixin import AddBreadcrumbOnShowMixin
from tribler.gui.utilities import connect, get_ui_file_path, tr
from tribler.gui.widgets.tablecontentmodel import SearchResultsModel
(widget_form, widget_class) = uic.loadUiType(get_ui_file_path('search_results.ui'))

def format_search_loading_label(search_request):
    if False:
        i = 10
        return i + 15
    data = {'total_peers': len(search_request.peers), 'num_complete_peers': len(search_request.peers_complete), 'num_remote_results': len(search_request.remote_results)}
    return tr('Remote responses: %(num_complete_peers)i / %(total_peers)i\nNew remote results received: %(num_remote_results)i') % data

@dataclass
class SearchRequest:
    uuid: uuid
    query: Query
    peers: set
    peers_complete: set = field(default_factory=set)
    remote_results: list = field(default_factory=list)

    @property
    def complete(self):
        if False:
            for i in range(10):
                print('nop')
        return self.peers == self.peers_complete

class SearchResultsWidget(AddBreadcrumbOnShowMixin, widget_form, widget_class):

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        widget_class.__init__(self, parent=parent)
        self._logger = logging.getLogger(self.__class__.__name__)
        try:
            self.setupUi(self)
        except SystemError:
            pass
        self.last_search_time = None
        self.last_search_query = None
        self.hide_xxx = None
        self.search_request = None
        connect(self.results_page_content.model_query_completed, self.on_local_query_completed)
        connect(self.search_progress_bar.ready_to_update_results, self.on_ready_to_update_results)

    def initialize(self, hide_xxx=False):
        if False:
            for i in range(10):
                print('nop')
        self.hide_xxx = hide_xxx
        self.results_page_content.initialize_content_page(hide_xxx=hide_xxx)
        self.results_page_content.channel_torrents_filter_input.setHidden(True)

    @property
    def has_results(self):
        if False:
            while True:
                i = 10
        return self.last_search_query is not None

    def check_can_show(self, query):
        if False:
            for i in range(10):
                print('nop')
        if self.last_search_query == query and self.last_search_time is not None and (time.time() - self.last_search_time < 1):
            self._logger.info('Same search query already sent within 500ms so dropping this one')
            return False
        return True

    def search(self, query: Query) -> bool:
        if False:
            i = 10
            return i + 15
        if not self.check_can_show(query.original_query):
            return False
        fts_query = to_fts_query(query.original_query)
        if not fts_query:
            return False
        self.last_search_query = query.original_query
        self.last_search_time = time.time()
        model = SearchResultsModel(endpoint_url='search', hide_xxx=self.results_page_content.hide_xxx, original_query=query.original_query, text_filter=to_fts_query(query.fts_text), tags=list(query.tags), type_filter=[REGULAR_TORRENT], exclude_deleted=True)
        self.results_page_content.initialize_root_model(model)
        self.setCurrentWidget(self.results_page)
        self.results_page_content.format_search_title()
        self.search_progress_bar.start()
        self.results_page_content.run_brain_dead_refresh()

        def register_request(response):
            if False:
                print('Hello World!')
            peers = set(response['peers'])
            self.search_request = SearchRequest(response['request_uuid'], query, peers)
            self.search_progress_bar.set_remote_total(len(peers))
        params = {'txt_filter': fts_query, 'hide_xxx': self.hide_xxx, 'tags': list(query.tags), 'metadata_type': REGULAR_TORRENT, 'exclude_deleted': True}
        request_manager.put('remote_query', register_request, url_params=params)
        return True

    def on_local_query_completed(self):
        if False:
            print('Hello World!')
        self.search_progress_bar.on_local_results()

    def reset(self):
        if False:
            while True:
                i = 10
        if self.currentWidget() == self.results_page:
            self.results_page_content.go_back_to_level(0)

    def update_loading_page(self, remote_results):
        if False:
            for i in range(10):
                print('nop')
        if not self.search_request or self.search_request.uuid != remote_results.get('uuid'):
            return
        peer = remote_results['peer']
        results = remote_results.get('results', [])
        self.search_request.peers_complete.add(peer)
        self.search_request.remote_results.append(results)
        new_items = self.results_page_content.model.add_remote_results(results)
        self.search_progress_bar.on_remote_results(len(new_items), len(self.search_request.peers_complete))

    def on_ready_to_update_results(self):
        if False:
            for i in range(10):
                print('nop')
        self.results_page_content.root_model.show_remote_results()