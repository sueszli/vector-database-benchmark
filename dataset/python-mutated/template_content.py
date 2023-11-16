from extras.plugins import PluginTemplateExtension

class SiteContent(PluginTemplateExtension):
    model = 'dcim.site'

    def left_page(self):
        if False:
            return 10
        return 'SITE CONTENT - LEFT PAGE'

    def right_page(self):
        if False:
            while True:
                i = 10
        return 'SITE CONTENT - RIGHT PAGE'

    def full_width_page(self):
        if False:
            return 10
        return 'SITE CONTENT - FULL WIDTH PAGE'

    def buttons(self):
        if False:
            i = 10
            return i + 15
        return 'SITE CONTENT - BUTTONS'

    def list_buttons(self):
        if False:
            i = 10
            return i + 15
        return 'SITE CONTENT - LIST BUTTONS'
template_extensions = [SiteContent]