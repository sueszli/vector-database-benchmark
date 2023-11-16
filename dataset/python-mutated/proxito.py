from readthedocs.settings.proxito.base import CommunityProxitoSettingsMixin
from .docker_compose import DockerBaseSettings

class ProxitoDevSettings(CommunityProxitoSettingsMixin, DockerBaseSettings):
    DONT_HIT_DB = False
    CACHEOPS_ENABLED = False

    @property
    def DEBUG_TOOLBAR_CONFIG(self):
        if False:
            while True:
                i = 10
        return {'SHOW_TOOLBAR_CALLBACK': lambda request: False}
ProxitoDevSettings.load_settings(__name__)