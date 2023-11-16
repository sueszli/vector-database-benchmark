from wagtail.admin.widgets.datetime import AdminDateTimeInput
from wagtail.models import Page
from .field_panel import FieldPanel
from .group import MultiFieldPanel

class PublishingPanel(MultiFieldPanel):

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        js_overlay_parent_selector = '#schedule-publishing-dialog'
        updated_kwargs = {'children': [FieldPanel('go_live_at', widget=AdminDateTimeInput(js_overlay_parent_selector=js_overlay_parent_selector, attrs={'data-controller': 'w-action', 'data-action': 'w-dialog:hidden->w-action#reset', 'data-w-dialog-target': 'notify'})), FieldPanel('expire_at', widget=AdminDateTimeInput(js_overlay_parent_selector=js_overlay_parent_selector, attrs={'data-controller': 'w-action', 'data-action': 'w-dialog:hidden->w-action#reset', 'data-w-dialog-target': 'notify'}))], 'classname': 'publishing'}
        updated_kwargs.update(kwargs)
        super().__init__(**updated_kwargs)

    @property
    def clean_name(self):
        if False:
            return 10
        return super().clean_name or 'publishing'

    class BoundPanel(MultiFieldPanel.BoundPanel):
        template_name = 'wagtailadmin/panels/publishing/schedule_publishing_panel.html'

        def get_context_data(self, parent_context=None):
            if False:
                while True:
                    i = 10
            context = super().get_context_data(parent_context)
            context['request'] = self.request
            context['instance'] = self.instance
            context['classname'] = self.classname
            context['model_opts'] = self.instance._meta
            if isinstance(self.instance, Page):
                context['page'] = self.instance
            return context

        def show_panel_furniture(self):
            if False:
                for i in range(10):
                    print('nop')
            return False