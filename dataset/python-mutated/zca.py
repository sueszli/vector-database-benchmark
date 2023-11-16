from pyramid.threadlocal import get_current_registry

class ZCAConfiguratorMixin:

    def hook_zca(self):
        if False:
            while True:
                i = 10
        "Call :func:`zope.component.getSiteManager.sethook` with the\n        argument :data:`pyramid.threadlocal.get_current_registry`, causing\n        the :term:`Zope Component Architecture` 'global' APIs such as\n        :func:`zope.component.getSiteManager`,\n        :func:`zope.component.getAdapter` and others to use the\n        :app:`Pyramid` :term:`application registry` rather than the Zope\n        'global' registry."
        from zope.component import getSiteManager
        getSiteManager.sethook(get_current_registry)

    def unhook_zca(self):
        if False:
            return 10
        'Call :func:`zope.component.getSiteManager.reset` to undo the\n        action of :meth:`pyramid.config.Configurator.hook_zca`.'
        from zope.component import getSiteManager
        getSiteManager.reset()