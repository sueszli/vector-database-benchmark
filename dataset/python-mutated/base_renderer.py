from abc import abstractmethod
from typing import runtime_checkable, Protocol
from feeluown.gui.widgets.tabbar import Tab, TabBar

class LibraryTabRendererMixin:

    def init_tabbar_signal_binding(self):
        if False:
            for i in range(10):
                print('nop')
        for (tab_id, signal) in self.get_tabid_signal_mapping().items():
            signal.connect(self.on_tab_id_activated(tab_id))

    def get_tabid_handler_mapping(self):
        if False:
            return 10
        return {Tab.songs: self.show_songs, Tab.albums: self.show_albums, Tab.artists: self.show_artists, Tab.playlists: self.show_playlists, Tab.videos: self.show_videos}

    def get_tabid_signal_mapping(self):
        if False:
            return 10
        return {Tab.songs: self.tabbar.show_songs_needed, Tab.albums: self.tabbar.show_albums_needed, Tab.artists: self.tabbar.show_artists_needed, Tab.playlists: self.tabbar.show_playlists_needed, Tab.videos: self.tabbar.show_videos_needed}

    def render_tabbar(self):
        if False:
            print('Hello World!')
        self.init_tabbar_signal_binding()
        self.tabbar.show()
        self.tabbar.library_mode()
        self.tabbar.check(self.tab_id)

    def on_tab_id_activated(self, tab_id):
        if False:
            print('Hello World!')

        def cb():
            if False:
                while True:
                    i = 10
            if tab_id != self.tab_id:
                self.show_by_tab_id(tab_id)
        return cb

    def show_by_tab_id(self, tab_id):
        if False:
            return 10
        raise NotImplementedError

class TabBarRendererMixin:
    """
    Requirements:
    1. the instance MUST has `tabs` attribute
    2. the instance MUST implement `render_by_tab_index`
    """

    def render_tab_bar(self):
        if False:
            i = 10
            return i + 15
        tab_bar = TabBar()
        ui = self._app.ui
        ui.toolbar.add_stacked_widget(tab_bar)
        ui.toolbar.set_top_stacked_widget(tab_bar)
        for tab in self.tabs:
            tab_bar.addTab(tab[0])
        tab_bar.setCurrentIndex(self.tab_index)
        tab_bar.tabBarClicked.connect(self.render_by_tab_index)

    def render_by_tab_index(self, tab_index):
        if False:
            print('Hello World!')
        raise NotImplementedError

@runtime_checkable
class VFillableBg(Protocol):
    """Protocol for widgets which has vertical fillable background

    This protocol indicates the background of the widget is transparent. The parent
    widget can fill color for the widget.

    .. versionadded:: 3.8.9
    """

    @abstractmethod
    def fillable_bg_height(self) -> int:
        if False:
            return 10
        raise NotImplementedError