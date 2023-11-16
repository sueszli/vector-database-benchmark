from pylons import config
from pylons import tmpl_context as c
from pylons import app_globals as g
from pylons.i18n import N_
from r2.lib.wrapped import Templated
from r2.lib.pages import LinkInfoBar, Reddit
from r2.lib.menus import NamedButton, NavButton, menu, NavMenu, OffsiteButton
from r2.lib.utils import timesince

def admin_menu(**kwargs):
    if False:
        print('Hello World!')
    buttons = [OffsiteButton('traffic', '/traffic'), NavButton(menu.awards, 'awards'), NavButton(menu.errors, 'error log')]
    admin_menu = NavMenu(buttons, title='admin tools', base_path='/admin', type='lightdrop', **kwargs)
    return admin_menu

class AdminSidebar(Templated):

    def __init__(self, user):
        if False:
            for i in range(10):
                print('nop')
        Templated.__init__(self)
        self.user = user

class SponsorSidebar(Templated):

    def __init__(self, user):
        if False:
            print('Hello World!')
        Templated.__init__(self)
        self.user = user

class Details(Templated):

    def __init__(self, link, *a, **kw):
        if False:
            for i in range(10):
                print('nop')
        Templated.__init__(self, *a, **kw)
        self.link = link

class AdminPage(Reddit):
    create_reddit_box = False
    submit_box = False
    extension_handling = False
    show_sidebar = False

    def __init__(self, nav_menus=None, *a, **kw):
        if False:
            return 10
        Reddit.__init__(self, *a, nav_menus=nav_menus, **kw)

class AdminProfileMenu(NavMenu):

    def __init__(self, path):
        if False:
            while True:
                i = 10
        NavMenu.__init__(self, [], base_path=path, title='admin', type='tabdrop')

class AdminLinkMenu(NavMenu):

    def __init__(self, link):
        if False:
            i = 10
            return i + 15
        NavMenu.__init__(self, [], title='admin', type='tabdrop')

class AdminNotesSidebar(Templated):
    EMPTY_MESSAGE = {'domain': N_('No notes for this domain'), 'ip': N_('No notes for this IP address'), 'subreddit': N_('No notes for this subreddit'), 'user': N_('No notes for this user')}
    SYSTEMS = {'domain': N_('domain'), 'ip': N_('IP address'), 'subreddit': N_('subreddit'), 'user': N_('user')}

    def __init__(self, system, subject):
        if False:
            for i in range(10):
                print('nop')
        from r2.models.admin_notes import AdminNotesBySystem
        self.system = system
        self.subject = subject
        self.author = c.user.name
        self.notes = AdminNotesBySystem.in_display_order(system, subject)
        for note in self.notes:
            note['timesince'] = timesince(note['when'])
        Templated.__init__(self)

class AdminLinkInfoBar(LinkInfoBar):
    pass

class AdminDetailsBar(Templated):
    pass
if config['r2.import_private']:
    from r2admin.lib.pages import *