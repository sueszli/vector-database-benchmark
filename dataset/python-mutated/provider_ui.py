from typing import TYPE_CHECKING
from feeluown.gui.provider_ui import AbstractProviderUi
from feeluown.utils import aio
from .ui import LibraryRenderer
from .provider import provider
if TYPE_CHECKING:
    from feeluown.app.gui_app import GuiApp

class LocalProviderUi(AbstractProviderUi):

    def __init__(self, app: 'GuiApp'):
        if False:
            i = 10
            return i + 15
        self._app = app

    @property
    def provider(self):
        if False:
            return 10
        return provider

    def register_pages(self, route):
        if False:
            return 10
        route('/local')(show_provider)

    def login_or_go_home(self):
        if False:
            i = 10
            return i + 15
        self._app.browser.goto(uri='/local')

def show_provider(req):
    if False:
        return 10
    if hasattr(req, 'ctx'):
        app: 'GuiApp' = req.ctx['app']
    else:
        app = req
    app.pl_uimgr.clear()
    app.ui.left_panel.my_music_con.hide()
    app.ui.left_panel.playlists_con.hide()
    aio.run_afn(app.ui.table_container.set_renderer, LibraryRenderer(provider.songs, provider.albums, provider.artists))