from django.utils.functional import cached_property
from django.utils.translation import gettext as _
from wagtail.admin.panels import Panel

class FormSubmissionsPanel(Panel):

    def on_model_bound(self):
        if False:
            print('Hello World!')
        if not self.heading:
            self.heading = _('%(model_name)s submissions') % {'model_name': self.model.get_verbose_name()}

    class BoundPanel(Panel.BoundPanel):
        template_name = 'wagtailforms/panels/form_responses_panel.html'

        @cached_property
        def submissions(self):
            if False:
                i = 10
                return i + 15
            form_page_model = self.panel.model
            form_submissions_model = form_page_model().get_submission_class()
            return form_submissions_model.objects.filter(page=self.instance)

        @cached_property
        def submission_count(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.submissions.count()

        def is_shown(self):
            if False:
                while True:
                    i = 10
            return self.submission_count

        def get_context_data(self, parent_context=None):
            if False:
                while True:
                    i = 10
            context = super().get_context_data(parent_context)
            context.update({'submission_count': self.submission_count, 'last_submit_time': self.submissions.order_by('submit_time').last().submit_time})
            return context