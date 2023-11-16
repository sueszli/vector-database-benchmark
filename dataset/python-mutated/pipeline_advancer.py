from django.contrib import messages
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.utils.translation import gettext_lazy as _
from sentry.api.utils import generate_organization_url
from sentry.identity.pipeline import IdentityProviderPipeline
from sentry.integrations.pipeline import IntegrationPipeline
from sentry.utils.http import absolute_uri, create_redirect_url
from sentry.web.decorators import transaction_start
from sentry.web.frontend.base import BaseView
PIPELINE_CLASSES = [IntegrationPipeline, IdentityProviderPipeline]
FORWARD_INSTALL_FOR = ['github']
from rest_framework.request import Request

class PipelineAdvancerView(BaseView):
    """Gets the current pipeline from the request and executes the current step."""
    auth_required = False
    csrf_protect = False

    @transaction_start('PipelineAdvancerView')
    def handle(self, request: Request, provider_id: str) -> HttpResponse:
        if False:
            for i in range(10):
                print('nop')
        pipeline = None
        for pipeline_cls in PIPELINE_CLASSES:
            pipeline = pipeline_cls.get_for_request(request=request)
            if pipeline:
                break
        if provider_id in FORWARD_INSTALL_FOR and request.GET.get('setup_action') == 'install' and (pipeline is None):
            installation_id = request.GET.get('installation_id')
            return self.redirect(reverse('integration-installation', args=[provider_id, installation_id]))
        if pipeline is None or not pipeline.is_valid():
            messages.add_message(request, messages.ERROR, _('Invalid request.'))
            return self.redirect('/')
        subdomain = pipeline.fetch_state('subdomain')
        if subdomain is not None and request.subdomain != subdomain:
            url_prefix = generate_organization_url(subdomain)
            redirect_url = absolute_uri(reverse('sentry-extension-setup', kwargs={'provider_id': provider_id}), url_prefix=url_prefix)
            return HttpResponseRedirect(create_redirect_url(request, redirect_url))
        response = pipeline.current_step()
        return response