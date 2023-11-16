class CMSApp:
    _urls = []
    _menus = []
    name = None
    app_name = None
    app_config = None
    permissions = True
    exclude_permissions = []

    def __new__(cls):
        if False:
            i = 10
            return i + 15
        '\n        We want to bind the CMSapp class to a specific AppHookConfig, but only one at a time\n        Checking for the runtime attribute should be a sane fix\n        '
        if cls.app_config:
            if getattr(cls.app_config, 'cmsapp', None) and cls.app_config.cmsapp != cls:
                raise RuntimeError('Only one AppHook per AppHookConfiguration must exists.\nAppHook %s already defined for %s AppHookConfig' % (cls.app_config.cmsapp.__name__, cls.app_config.__name__))
            cls.app_config.cmsapp = cls
        return super(CMSApp, cls).__new__(cls)

    def get_configs(self):
        if False:
            return 10
        '\n        Returns all the apphook configuration instances.\n        '
        raise NotImplementedError('Configurable AppHooks must implement this method')

    def get_config(self, namespace):
        if False:
            while True:
                i = 10
        '\n        Returns the apphook configuration instance linked to the given namespace\n        '
        raise NotImplementedError('Configurable AppHooks must implement this method')

    def get_config_add_url(self):
        if False:
            while True:
                i = 10
        '\n        Returns the url to add a new apphook configuration instance\n        (usually the model admin add view)\n        '
        raise NotImplementedError('Configurable AppHooks must implement this method')

    def get_menus(self, page=None, language=None, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Returns the menus for the apphook instance, eventually selected according\n        to the given arguments.\n\n        By default it returns the menus assigned to :py:attr:`CMSApp._menus`.\n\n        If no menus are returned, then the user will need to attach menus to pages\n        manually in the admin.\n\n        This method must return all the menus used by this apphook if no arguments are\n        provided. Example::\n\n            if page and page.reverse_id == 'page1':\n                return [Menu1]\n            elif page and page.reverse_id == 'page2':\n                return [Menu2]\n            else:\n                return [Menu1, Menu2]\n\n        :param page: page the apphook is attached to\n        :param language: current site language\n        :return: list of menu classes\n        "
        return self._menus

    def get_urls(self, page=None, language=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Returns the urlconfs for the apphook instance, eventually selected\n        according to the given arguments.\n\n        By default it returns the urls assigned to :py:attr:`CMSApp._urls`\n\n        This method **must** return a non empty list of urlconfs,\n        even if no argument is passed.\n\n        :param page: page the apphook is attached to\n        :param language: current site language\n        :return: list of urlconfs strings\n        '
        return self._urls