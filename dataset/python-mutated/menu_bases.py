from django.core.exceptions import ValidationError
from django.db.models import Q
from cms.apphook_pool import apphook_pool
from cms.models import Page
from menus.base import Menu

class CMSAttachMenu(Menu):
    cms_enabled = True
    instance = None
    name = None

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        if self.cms_enabled and (not self.name):
            raise ValidationError('the menu %s is a CMSAttachMenu but has no name defined!' % self.__class__.__name__)

    @classmethod
    def get_apphooks(cls):
        if False:
            print('Hello World!')
        '\n        Returns a list of apphooks to which this CMSAttachMenu is attached.\n\n        Calling this does NOT produce DB queries.\n        '
        apps = []
        for (key, _) in apphook_pool.get_apphooks():
            app = apphook_pool.get_apphook(key)
            if cls in app.get_menus():
                apps.append(app)
        return apps

    @classmethod
    def get_instances(cls):
        if False:
            return 10
        '\n        Return a list (queryset, really) of all CMS Page objects (in this case)\n        that are currently using this CMSAttachMenu either directly as a\n        navigation_extender, or, as part of an apphook.\n\n        Calling this DOES perform a DB query.\n        '
        parent_apps = []
        for app in cls.get_apphooks():
            parent_apps.append(app.__class__.__name__)
        return Page.objects.filter(Q(application_urls__in=parent_apps) | Q(navigation_extenders=cls.__name__))