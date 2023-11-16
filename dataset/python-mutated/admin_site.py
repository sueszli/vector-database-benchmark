from django.contrib.admin import AdminSite
from django.contrib.admin.models import LogEntry
from django.contrib.sites.admin import SiteAdmin
from django.contrib.sites.models import Site
from accounts.admin import *
from blog.admin import *
from blog.models import *
from comments.admin import *
from comments.models import *
from djangoblog.logentryadmin import LogEntryAdmin
from oauth.admin import *
from oauth.models import *
from owntracks.admin import *
from owntracks.models import *
from servermanager.admin import *
from servermanager.models import *

class DjangoBlogAdminSite(AdminSite):
    site_header = 'djangoblog administration'
    site_title = 'djangoblog site admin'

    def __init__(self, name='admin'):
        if False:
            i = 10
            return i + 15
        super().__init__(name)

    def has_permission(self, request):
        if False:
            i = 10
            return i + 15
        return request.user.is_superuser
admin_site = DjangoBlogAdminSite(name='admin')
admin_site.register(Article, ArticlelAdmin)
admin_site.register(Category, CategoryAdmin)
admin_site.register(Tag, TagAdmin)
admin_site.register(Links, LinksAdmin)
admin_site.register(SideBar, SideBarAdmin)
admin_site.register(BlogSettings, BlogSettingsAdmin)
admin_site.register(commands, CommandsAdmin)
admin_site.register(EmailSendLog, EmailSendLogAdmin)
admin_site.register(BlogUser, BlogUserAdmin)
admin_site.register(Comment, CommentAdmin)
admin_site.register(OAuthUser, OAuthUserAdmin)
admin_site.register(OAuthConfig, OAuthConfigAdmin)
admin_site.register(OwnTrackLog, OwnTrackLogsAdmin)
admin_site.register(Site, SiteAdmin)
admin_site.register(LogEntry, LogEntryAdmin)