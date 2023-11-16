from .base import Panel

class HelpPanel(Panel):
    """
    A panel to display helpful information to the user.

    This panel does not support the ``help_text`` parameter.
    """

    def __init__(self, content='', template='wagtailadmin/panels/help_panel.html', **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.content = content
        self.template = template

    def clone_kwargs(self):
        if False:
            return 10
        kwargs = super().clone_kwargs()
        del kwargs['help_text']
        kwargs.update(content=self.content, template=self.template)
        return kwargs

    @property
    def clean_name(self):
        if False:
            while True:
                i = 10
        return super().clean_name or 'help'

    class BoundPanel(Panel.BoundPanel):

        def __init__(self, **kwargs):
            if False:
                i = 10
                return i + 15
            super().__init__(**kwargs)
            self.template_name = self.panel.template
            self.content = self.panel.content