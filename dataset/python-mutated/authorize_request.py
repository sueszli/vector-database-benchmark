from lxml import etree, objectify
from urllib2 import urlopen, Request
from StringIO import StringIO
import xml.etree.ElementTree as ET
from uuid import uuid4
from odoo import _
from odoo.exceptions import ValidationError, UserError
from odoo import _
XMLNS = 'AnetApi/xml/v1/schema/AnetApiSchema.xsd'

def strip_ns(xml, ns):
    if False:
        print('Hello World!')
    'Strip the provided name from tag names.\n\n    :param str xml: xml document\n    :param str ns: namespace to strip\n\n    :rtype: etree._Element\n    :return: the parsed xml string with the namespace prefix removed\n    '
    it = ET.iterparse(StringIO(xml))
    ns_prefix = '{%s}' % XMLNS
    for (_, el) in it:
        if el.tag.startswith(ns_prefix):
            el.tag = el.tag[len(ns_prefix):]
    return it.root

class AuthorizeAPI:
    """Authorize.net Gateway API integration.

    This class allows contacting the Authorize.net API with simple operation
    requests. It implements a *very limited* subset of the complete API
    (http://developer.authorize.net/api/reference); namely:
        - Customer Profile/Payment Profile creation
        - Transaction authorization/capture/voiding
    """

    def __init__(self, acquirer):
        if False:
            for i in range(10):
                print('nop')
        'Initiate the environment with the acquirer data.\n\n        :param record acquirer: payment.acquirer account that will be contacted\n        '
        if acquirer.environment == 'test':
            self.url = 'https://apitest.authorize.net/xml/v1/request.api'
        else:
            self.url = 'https://api.authorize.net/xml/v1/request.api'
        self.name = acquirer.authorize_login
        self.transaction_key = acquirer.authorize_transaction_key

    def _authorize_request(self, data):
        if False:
            return 10
        'Encode, send and process the request to the Authorize.net API.\n\n        Encodes the xml data and process the response. Note that only a basic\n        processing is done at this level (namespace cleanup, basic error management).\n\n        :param etree._Element data: etree data to process\n        '
        data = etree.tostring(data, xml_declaration=True, encoding='utf-8')
        request = Request(self.url, data)
        request.add_header('Content-Type', 'text/xml')
        response = urlopen(request).read()
        response = strip_ns(response, XMLNS)
        if response.find('messages/resultCode').text == 'Error':
            messages = map(lambda m: m.text, response.findall('messages/message/text'))
            raise ValidationError(_('Authorize.net Error Message(s):\n %s') % '\n'.join(messages))
        return response

    def _base_tree(self, requestType):
        if False:
            return 10
        'Create a basic tree containing authentication information.\n\n        Create a etree Element of type requestType and appends the Authorize.net\n        credentials (they are always required).\n        :param str requestType: the type of request to send to Authorize.net\n                                See http://developer.authorize.net/api/reference\n                                for available types.\n        :return: basic etree Element of the requested type\n                               containing credentials information\n        :rtype: etree._Element\n        '
        root = etree.Element(requestType, xmlns=XMLNS)
        auth = etree.SubElement(root, 'merchantAuthentication')
        etree.SubElement(auth, 'name').text = self.name
        etree.SubElement(auth, 'transactionKey').text = self.transaction_key
        return root

    def create_customer_profile(self, partner, cardnumber, expiration_date, card_code):
        if False:
            while True:
                i = 10
        "Create a payment and customer profile in the Authorize.net backend.\n\n        Creates a customer profile for the partner/credit card combination and links\n        a corresponding payment profile to it. Note that a single partner in the Odoo\n        database can have multiple customer profiles in Authorize.net (i.e. a customer\n        profile is created for every res.partner/payment.token couple).\n\n        :param record partner: the res.partner record of the customer\n        :param str cardnumber: cardnumber in string format (numbers only, no separator)\n        :param str expiration_date: expiration date in 'YYYY-MM' string format\n        :param str card_code: three- or four-digit verification number\n\n        :return: a dict containing the profile_id and payment_profile_id of the\n                 newly created customer profile and payment profile\n        :rtype: dict\n        "
        root = self._base_tree('createCustomerProfileRequest')
        profile = etree.SubElement(root, 'profile')
        etree.SubElement(profile, 'merchantCustomerId').text = 'ODOO-%s-%s' % (partner.id, uuid4().hex[:8])
        etree.SubElement(profile, 'email').text = partner.email
        payment_profile = etree.SubElement(profile, 'paymentProfiles')
        etree.SubElement(payment_profile, 'customerType').text = 'business' if partner.is_company else 'individual'
        billTo = etree.SubElement(payment_profile, 'billTo')
        etree.SubElement(billTo, 'address').text = partner.street + (partner.street2 if partner.street2 else '') or None
        etree.SubElement(billTo, 'city').text = partner.city
        etree.SubElement(billTo, 'state').text = partner.state_id.name or None
        etree.SubElement(billTo, 'zip').text = partner.zip
        etree.SubElement(billTo, 'country').text = partner.country_id.name or None
        payment = etree.SubElement(payment_profile, 'payment')
        creditCard = etree.SubElement(payment, 'creditCard')
        etree.SubElement(creditCard, 'cardNumber').text = cardnumber
        etree.SubElement(creditCard, 'expirationDate').text = expiration_date
        etree.SubElement(creditCard, 'cardCode').text = card_code
        etree.SubElement(root, 'validationMode').text = 'liveMode'
        response = self._authorize_request(root)
        res = dict()
        res['profile_id'] = response.find('customerProfileId').text
        res['payment_profile_id'] = response.find('customerPaymentProfileIdList/numericString').text
        return res

    def create_customer_profile_from_tx(self, partner, transaction_id):
        if False:
            return 10
        'Create an Auth.net payment/customer profile from an existing transaction.\n\n        Creates a customer profile for the partner/credit card combination and links\n        a corresponding payment profile to it. Note that a single partner in the Odoo\n        database can have multiple customer profiles in Authorize.net (i.e. a customer\n        profile is created for every res.partner/payment.token couple).\n\n        Note that this function makes 2 calls to the authorize api, since we need to\n        obtain a partial cardnumber to generate a meaningful payment.token name.\n\n        :param record partner: the res.partner record of the customer\n        :param str transaction_id: id of the authorized transaction in the\n                                   Authorize.net backend\n\n        :return: a dict containing the profile_id and payment_profile_id of the\n                 newly created customer profile and payment profile as well as the\n                 last digits of the card number\n        :rtype: dict\n        '
        root = self._base_tree('createCustomerProfileFromTransactionRequest')
        etree.SubElement(root, 'transId').text = transaction_id
        customer = etree.SubElement(root, 'customer')
        etree.SubElement(customer, 'merchantCustomerId').text = 'ODOO-%s-%s' % (partner.id, uuid4().hex[:8])
        etree.SubElement(customer, 'email').text = partner.email or ''
        response = self._authorize_request(root)
        res = dict()
        res['profile_id'] = response.find('customerProfileId').text
        res['payment_profile_id'] = response.find('customerPaymentProfileIdList/numericString').text
        root_profile = self._base_tree('getCustomerPaymentProfileRequest')
        etree.SubElement(root_profile, 'customerProfileId').text = res['profile_id']
        etree.SubElement(root_profile, 'customerPaymentProfileId').text = res['payment_profile_id']
        response_profile = self._authorize_request(root_profile)
        res['name'] = response_profile.find('paymentProfile/payment/creditCard/cardNumber').text
        return res

    def auth_and_capture(self, token, amount, reference):
        if False:
            for i in range(10):
                print('nop')
        'Authorize and capture a payment for the given amount.\n\n        Authorize and immediately capture a payment for the given payment.token\n        record for the specified amount with reference as communication.\n\n        :param record token: the payment.token record that must be charged\n        :param str amount: transaction amount (up to 15 digits with decimal point)\n        :param str reference: used as "invoiceNumber" in the Authorize.net backend\n\n        :return: a dict containing the response code, transaction id and transaction type\n        :rtype: dict\n        '
        root = self._base_tree('createTransactionRequest')
        tx = etree.SubElement(root, 'transactionRequest')
        etree.SubElement(tx, 'transactionType').text = 'authCaptureTransaction'
        etree.SubElement(tx, 'amount').text = str(amount)
        profile = etree.SubElement(tx, 'profile')
        etree.SubElement(profile, 'customerProfileId').text = token.authorize_profile
        payment_profile = etree.SubElement(profile, 'paymentProfile')
        etree.SubElement(payment_profile, 'paymentProfileId').text = token.acquirer_ref
        order = etree.SubElement(tx, 'order')
        etree.SubElement(order, 'invoiceNumber').text = reference
        response = self._authorize_request(root)
        res = dict()
        res['x_response_code'] = response.find('transactionResponse/responseCode').text
        res['x_trans_id'] = response.find('transactionResponse/transId').text
        res['x_type'] = 'auth_capture'
        return res

    def authorize(self, token, amount, reference):
        if False:
            for i in range(10):
                print('nop')
        'Authorize a payment for the given amount.\n\n        Authorize (without capture) a payment for the given payment.token\n        record for the specified amount with reference as communication.\n\n        :param record token: the payment.token record that must be charged\n        :param str amount: transaction amount (up to 15 digits with decimal point)\n        :param str reference: used as "invoiceNumber" in the Authorize.net backend\n\n        :return: a dict containing the response code, transaction id and transaction type\n        :rtype: dict\n        '
        root = self._base_tree('createTransactionRequest')
        tx = etree.SubElement(root, 'transactionRequest')
        etree.SubElement(tx, 'transactionType').text = 'authOnlyTransaction'
        etree.SubElement(tx, 'amount').text = str(amount)
        profile = etree.SubElement(tx, 'profile')
        etree.SubElement(profile, 'customerProfileId').text = token.authorize_profile
        payment_profile = etree.SubElement(profile, 'paymentProfile')
        etree.SubElement(payment_profile, 'paymentProfileId').text = token.acquirer_ref
        order = etree.SubElement(tx, 'order')
        etree.SubElement(order, 'invoiceNumber').text = reference
        response = self._authorize_request(root)
        res = dict()
        res['x_response_code'] = response.find('transactionResponse/responseCode').text
        res['x_trans_id'] = response.find('transactionResponse/transId').text
        res['x_type'] = 'auth_only'
        return res

    def capture(self, transaction_id, amount):
        if False:
            for i in range(10):
                print('nop')
        'Capture a previously authorized payment for the given amount.\n\n        Capture a previsouly authorized payment. Note that the amount is required\n        even though we do not support partial capture.\n\n        :param str transaction_id: id of the authorized transaction in the\n                                   Authorize.net backend\n        :param str amount: transaction amount (up to 15 digits with decimal point)\n\n        :return: a dict containing the response code, transaction id and transaction type\n        :rtype: dict\n        '
        root = self._base_tree('createTransactionRequest')
        tx = etree.SubElement(root, 'transactionRequest')
        etree.SubElement(tx, 'transactionType').text = 'priorAuthCaptureTransaction'
        etree.SubElement(tx, 'amount').text = str(amount)
        etree.SubElement(tx, 'refTransId').text = transaction_id
        response = self._authorize_request(root)
        res = dict()
        res['x_response_code'] = response.find('transactionResponse/responseCode').text
        res['x_trans_id'] = response.find('transactionResponse/transId').text
        res['x_type'] = 'prior_auth_capture'
        return res

    def void(self, transaction_id):
        if False:
            return 10
        'Void a previously authorized payment.\n\n        :param str transaction_id: the id of the authorized transaction in the\n                                   Authorize.net backend\n\n        :return: a dict containing the response code, transaction id and transaction type\n        :rtype: dict\n        '
        root = self._base_tree('createTransactionRequest')
        tx = etree.SubElement(root, 'transactionRequest')
        etree.SubElement(tx, 'transactionType').text = 'voidTransaction'
        etree.SubElement(tx, 'refTransId').text = transaction_id
        response = self._authorize_request(root)
        res = dict()
        res['x_response_code'] = response.find('transactionResponse/responseCode').text
        res['x_trans_id'] = response.find('transactionResponse/transId').text
        res['x_type'] = 'void'
        return res

    def test_authenticate(self):
        if False:
            i = 10
            return i + 15
        'Test Authorize.net communication with a simple credentials check.\n\n        :return: True if authentication was successful, else False (or throws an error)\n        :rtype: bool\n        '
        test_auth = self._base_tree('authenticateTestRequest')
        response = self._authorize_request(test_auth)
        root = objectify.fromstring(response)
        if root.find('{ns}messages/{ns}resultCode'.format(ns='{%s}' % XMLNS)) == 'Ok':
            return True
        return False