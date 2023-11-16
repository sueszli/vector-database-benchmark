from sentry.utils.signing import unsign
from sentry.web.frontend.base import control_silo_view
from .integration_extension_configuration import IntegrationExtensionConfigurationView
INSTALL_EXPIRATION_TIME = 60 * 60 * 24

@control_silo_view
class MsTeamsExtensionConfigurationView(IntegrationExtensionConfigurationView):
    provider = 'msteams'
    external_provider_key = 'msteams'

    def map_params_to_state(self, params):
        if False:
            while True:
                i = 10
        params = params.copy()
        signed_params = params['signed_params']
        del params['signed_params']
        params.update(unsign(signed_params, max_age=INSTALL_EXPIRATION_TIME))
        return params