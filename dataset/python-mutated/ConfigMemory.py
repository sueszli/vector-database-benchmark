from .ConfigBase import ConfigBase
from ..AppriseLocale import gettext_lazy as _

class ConfigMemory(ConfigBase):
    """
    For information that was loaded from memory and does not
    persist anywhere.
    """
    service_name = _('Memory')
    protocol = 'memory'

    def __init__(self, content, **kwargs):
        if False:
            return 10
        "\n        Initialize Memory Object\n\n        Memory objects just store the raw configuration in memory.  There is\n        no external reference point. It's always considered cached.\n        "
        super().__init__(**kwargs)
        self.content = content
        if self.config_format is None:
            self.config_format = ConfigMemory.detect_config_format(self.content)
        return

    def url(self, privacy=False, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        return 'memory://'

    def read(self, **kwargs):
        if False:
            return 10
        '\n        Simply return content stored into memory\n        '
        return self.content

    @staticmethod
    def parse_url(url):
        if False:
            return 10
        '\n        Memory objects have no parseable URL\n\n        '
        return None