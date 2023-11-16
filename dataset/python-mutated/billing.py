import random
from string import ascii_letters, digits
from pyramid.httpexceptions import HTTPNotFound, HTTPSeeOther
from pyramid.view import view_config, view_defaults
from warehouse.api.billing import handle_billing_webhook_event
from warehouse.organizations.models import Organization
from warehouse.subscriptions.interfaces import IBillingService
from warehouse.subscriptions.services import MockStripeBillingService

@view_defaults(context=Organization, uses_session=True, require_methods=False, permission='manage:billing', has_translations=True, require_reauth=True)
class MockBillingViews:

    def __init__(self, organization, request):
        if False:
            while True:
                i = 10
        billing_service = request.find_service(IBillingService, context=None)
        if not request.organization_access or not isinstance(billing_service, MockStripeBillingService):
            raise HTTPNotFound
        self.organization = organization
        self.request = request

    @view_config(route_name='mock.billing.checkout-session', renderer='mock/billing/checkout-session.html')
    def mock_checkout_session(self):
        if False:
            for i in range(10):
                print('nop')
        return {'organization': self.organization}

    @view_config(route_name='mock.billing.portal-session', renderer='mock/billing/portal-session.html')
    def mock_portal_session(self):
        if False:
            return 10
        return {'organization': self.organization}

    @view_config(route_name='mock.billing.trigger-checkout-session-completed')
    def mock_trigger_checkout_session_completed(self):
        if False:
            print('Hello World!')
        mock_event = {'type': 'checkout.session.completed', 'data': {'object': {'id': 'mockcs_' + ''.join(random.choices(digits + ascii_letters, k=58)), 'customer': self.organization.customer and self.organization.customer.customer_id, 'customer_email': self.organization.customer and self.organization.customer.billing_email, 'status': 'complete', 'subscription': 'mocksub_' + ''.join(random.choices(digits + ascii_letters, k=24))}}}
        handle_billing_webhook_event(self.request, mock_event)
        return HTTPSeeOther(self.request.route_path('manage.organizations'))