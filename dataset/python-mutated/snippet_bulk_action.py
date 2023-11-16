from django.utils.functional import classproperty
from wagtail.admin.admin_url_finder import AdminURLFinder
from wagtail.admin.views.bulk_action import BulkAction
from wagtail.snippets.models import get_snippet_models

class SnippetBulkAction(BulkAction):

    @classproperty
    def models(cls):
        if False:
            while True:
                i = 10
        return get_snippet_models()

    def object_context(self, snippet):
        if False:
            i = 10
            return i + 15
        return {'item': snippet, 'edit_url': AdminURLFinder(self.request.user).get_edit_url(snippet)}

    def get_context_data(self, **kwargs):
        if False:
            while True:
                i = 10
        kwargs.update({'model_opts': self.model._meta, 'header_icon': self.model.snippet_viewset.icon})
        return super().get_context_data(**kwargs)

    def get_execution_context(self):
        if False:
            for i in range(10):
                print('nop')
        return {**super().get_execution_context(), 'self': self}