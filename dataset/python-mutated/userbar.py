from django.template.loader import render_to_string
from django.utils.translation import gettext_lazy as _

class BaseItem:
    template = 'wagtailadmin/userbar/item_base.html'

    def get_context_data(self, request):
        if False:
            while True:
                i = 10
        return {'self': self, 'request': request}

    def render(self, request):
        if False:
            for i in range(10):
                print('nop')
        return render_to_string(self.template, self.get_context_data(request), request=request)

class AdminItem(BaseItem):
    template = 'wagtailadmin/userbar/item_admin.html'

    def render(self, request):
        if False:
            i = 10
            return i + 15
        if not request.user.has_perm('wagtailadmin.access_admin'):
            return ''
        return super().render(request)

class AccessibilityItem(BaseItem):
    """A userbar item that runs the accessibility checker."""
    template = 'wagtailadmin/userbar/item_accessibility.html'
    axe_include = ['body']
    axe_exclude = []
    _axe_default_exclude = [{'fromShadowDOM': ['wagtail-userbar']}]
    axe_run_only = ['button-name', 'empty-heading', 'empty-table-header', 'frame-title', 'heading-order', 'input-button-name', 'link-name', 'p-as-heading']
    axe_rules = {}
    axe_messages = {'button-name': _('Button text is empty. Use meaningful text for screen reader users.'), 'empty-heading': _('Empty heading found. Use meaningful text for screen reader users.'), 'empty-table-header': _('Table header text is empty. Use meaningful text for screen reader users.'), 'frame-title': _('Empty frame title found. Use a meaningful title for screen reader users.'), 'heading-order': _('Incorrect heading hierarchy. Avoid skipping levels.'), 'input-button-name': _('Input button text is empty. Use meaningful text for screen reader users.'), 'link-name': _('Link text is empty. Use meaningful text for screen reader users.'), 'p-as-heading': _('Misusing paragraphs as headings. Use proper heading tags.')}

    def get_axe_include(self, request):
        if False:
            i = 10
            return i + 15
        'Returns a list of CSS selector(s) to test specific parts of the page.'
        return self.axe_include

    def get_axe_exclude(self, request):
        if False:
            while True:
                i = 10
        'Returns a list of CSS selector(s) to exclude specific parts of the page from testing.'
        return self.axe_exclude + self._axe_default_exclude

    def get_axe_run_only(self, request):
        if False:
            print('Hello World!')
        'Returns a list of axe-core tags or a list of axe-core rule IDs (not a mix of both).'
        return self.axe_run_only

    def get_axe_rules(self, request):
        if False:
            print('Hello World!')
        'Returns a dictionary that maps axe-core rule IDs to a dictionary of rule options.'
        return self.axe_rules

    def get_axe_messages(self, request):
        if False:
            i = 10
            return i + 15
        'Returns a dictionary that maps axe-core rule IDs to custom translatable strings.'
        return self.axe_messages

    def get_axe_context(self, request):
        if False:
            return 10
        '\n        Returns the `context object <https://github.com/dequelabs/axe-core/blob/develop/doc/context.md>`_\n        to be passed as the\n        `context parameter <https://github.com/dequelabs/axe-core/blob/develop/doc/API.md#context-parameter>`_\n        for ``axe.run``.\n        '
        return {'include': self.get_axe_include(request), 'exclude': self.get_axe_exclude(request)}

    def get_axe_options(self, request):
        if False:
            i = 10
            return i + 15
        '\n        Returns the options object to be passed as the\n        `options parameter <https://github.com/dequelabs/axe-core/blob/develop/doc/API.md#options-parameter>`_\n        for ``axe.run``.\n        '
        options = {'runOnly': self.get_axe_run_only(request), 'rules': self.get_axe_rules(request)}
        if not options['runOnly']:
            options.pop('runOnly')
        return options

    def get_axe_configuration(self, request):
        if False:
            while True:
                i = 10
        return {'context': self.get_axe_context(request), 'options': self.get_axe_options(request), 'messages': self.get_axe_messages(request)}

    def get_context_data(self, request):
        if False:
            i = 10
            return i + 15
        return {**super().get_context_data(request), 'axe_configuration': self.get_axe_configuration(request)}

    def render(self, request):
        if False:
            for i in range(10):
                print('nop')
        if not request.user.has_perm('wagtailadmin.access_admin'):
            return ''
        return super().render(request)

class AddPageItem(BaseItem):
    template = 'wagtailadmin/userbar/item_page_add.html'

    def __init__(self, page):
        if False:
            i = 10
            return i + 15
        self.page = page
        self.parent_page = page.get_parent()

    def render(self, request):
        if False:
            i = 10
            return i + 15
        if not self.page.id:
            return ''
        if not request.user.has_perm('wagtailadmin.access_admin'):
            return ''
        permission_checker = self.page.permissions_for_user(request.user)
        if not permission_checker.can_add_subpage():
            return ''
        return super().render(request)

class ExplorePageItem(BaseItem):
    template = 'wagtailadmin/userbar/item_page_explore.html'

    def __init__(self, page):
        if False:
            i = 10
            return i + 15
        self.page = page
        self.parent_page = page.get_parent()

    def render(self, request):
        if False:
            while True:
                i = 10
        if not self.page.id:
            return ''
        if not request.user.has_perm('wagtailadmin.access_admin'):
            return ''
        permission_checker = self.parent_page.permissions_for_user(request.user)
        if not permission_checker.can_edit() and (not permission_checker.can_publish_subpage()):
            return ''
        return super().render(request)

class EditPageItem(BaseItem):
    template = 'wagtailadmin/userbar/item_page_edit.html'

    def __init__(self, page):
        if False:
            while True:
                i = 10
        self.page = page

    def render(self, request):
        if False:
            i = 10
            return i + 15
        if not self.page.id:
            return ''
        try:
            if request.is_preview and request.is_editing:
                return ''
        except AttributeError:
            pass
        if not request.user.has_perm('wagtailadmin.access_admin'):
            return ''
        permission_checker = self.page.permissions_for_user(request.user)
        if not permission_checker.can_edit():
            return ''
        return super().render(request)