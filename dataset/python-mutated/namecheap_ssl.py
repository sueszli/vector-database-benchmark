"""
Namecheap SSL Certificate Management

.. versionadded:: 2017.7.0

Prerequisites
-------------

This module uses the ``requests`` Python module to communicate to the namecheap
API.

Configuration
-------------

The Namecheap username, API key and URL should be set in the minion configuration
file, or in the Pillar data.

.. code-block:: yaml

    namecheap.name: companyname
    namecheap.key: a1b2c3d4e5f67a8b9c0d1e2f3
    namecheap.client_ip: 162.155.30.172
    #Real url
    namecheap.url: https://api.namecheap.com/xml.response
    #Sandbox url
    #namecheap.url: https://api.sandbox.namecheap.xml.response
"""
import logging
import salt.utils.files
import salt.utils.stringutils
try:
    import salt.utils.namecheap
    CAN_USE_NAMECHEAP = True
except ImportError:
    CAN_USE_NAMECHEAP = False
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Check to make sure requests and xml are installed and requests\n    '
    if CAN_USE_NAMECHEAP:
        return 'namecheap_ssl'
    return False

def reissue(csr_file, certificate_id, web_server_type, approver_email=None, http_dc_validation=False, **kwargs):
    if False:
        print('Hello World!')
    "\n    Reissues a purchased SSL certificate. Returns a dictionary of result\n    values.\n\n    csr_file\n        Path to Certificate Signing Request file\n\n    certificate_id\n        Unique ID of the SSL certificate you wish to activate\n\n    web_server_type\n        The type of certificate format to return. Possible values include:\n\n        - apache2\n        - apacheapachessl\n        - apacheopenssl\n        - apacheraven\n        - apachessl\n        - apachessleay\n        - c2net\n        - cobaltseries\n        - cpanel\n        - domino\n        - dominogo4625\n        - dominogo4626\n        - ensim\n        - hsphere\n        - ibmhttp\n        - iis\n        - iis4\n        - iis5\n        - iplanet\n        - ipswitch\n        - netscape\n        - other\n        - plesk\n        - tomcat\n        - weblogic\n        - website\n        - webstar\n        - zeusv3\n\n    approver_email\n        The email ID which is on the approver email list.\n\n        .. note::\n            ``http_dc_validation`` must be set to ``False`` if this option is\n            used.\n\n    http_dc_validation : False\n        Whether or not to activate using HTTP-based validation.\n\n    .. note::\n        For other parameters which may be required, see here__.\n\n        .. __: https://www.namecheap.com/support/api/methods/ssl/reissue.aspx\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' namecheap_ssl.reissue my-csr-file my-cert-id apachessl\n    "
    return __get_certificates('namecheap.ssl.reissue', 'SSLReissueResult', csr_file, certificate_id, web_server_type, approver_email, http_dc_validation, kwargs)

def activate(csr_file, certificate_id, web_server_type, approver_email=None, http_dc_validation=False, **kwargs):
    if False:
        return 10
    "\n    Activates a newly-purchased SSL certificate. Returns a dictionary of result\n    values.\n\n    csr_file\n        Path to Certificate Signing Request file\n\n    certificate_id\n        Unique ID of the SSL certificate you wish to activate\n\n    web_server_type\n        The type of certificate format to return. Possible values include:\n\n        - apache2\n        - apacheapachessl\n        - apacheopenssl\n        - apacheraven\n        - apachessl\n        - apachessleay\n        - c2net\n        - cobaltseries\n        - cpanel\n        - domino\n        - dominogo4625\n        - dominogo4626\n        - ensim\n        - hsphere\n        - ibmhttp\n        - iis\n        - iis4\n        - iis5\n        - iplanet\n        - ipswitch\n        - netscape\n        - other\n        - plesk\n        - tomcat\n        - weblogic\n        - website\n        - webstar\n        - zeusv3\n\n    approver_email\n        The email ID which is on the approver email list.\n\n        .. note::\n            ``http_dc_validation`` must be set to ``False`` if this option is\n            used.\n\n    http_dc_validation : False\n        Whether or not to activate using HTTP-based validation.\n\n    .. note::\n        For other parameters which may be required, see here__.\n\n        .. __: https://www.namecheap.com/support/api/methods/ssl/activate.aspx\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' namecheap_ssl.activate my-csr-file my-cert-id apachessl\n    "
    return __get_certificates('namecheap.ssl.activate', 'SSLActivateResult', csr_file, certificate_id, web_server_type, approver_email, http_dc_validation, kwargs)

def __get_certificates(command, result_tag_name, csr_file, certificate_id, web_server_type, approver_email, http_dc_validation, kwargs):
    if False:
        for i in range(10):
            print('nop')
    web_server_types = ('apacheopenssl', 'apachessl', 'apacheraven', 'apachessleay', 'c2net', 'ibmhttp', 'iplanet', 'domino', 'dominogo4625', 'dominogo4626', 'netscape', 'zeusv3', 'apache2', 'apacheapachessl', 'cobaltseries', 'cpanel', 'ensim', 'hsphere', 'ipswitch', 'plesk', 'tomcat', 'weblogic', 'website', 'webstar', 'iis', 'other', 'iis4', 'iis5')
    if web_server_type not in web_server_types:
        log.error('Invalid option for web_server_type=%s', web_server_type)
        raise Exception('Invalid option for web_server_type=' + web_server_type)
    if approver_email is not None and http_dc_validation:
        log.error('approver_email and http_dc_validation cannot both have values')
        raise Exception('approver_email and http_dc_validation cannot both have values')
    if approver_email is None and (not http_dc_validation):
        log.error('approver_email or http_dc_validation must have a value')
        raise Exception('approver_email or http_dc_validation must have a value')
    opts = salt.utils.namecheap.get_opts(command)
    with salt.utils.files.fopen(csr_file, 'rb') as csr_handle:
        opts['csr'] = salt.utils.stringutils.to_unicode(csr_handle.read())
    opts['CertificateID'] = certificate_id
    opts['WebServerType'] = web_server_type
    if approver_email is not None:
        opts['ApproverEmail'] = approver_email
    if http_dc_validation:
        opts['HTTPDCValidation'] = 'True'
    for (key, value) in kwargs.items():
        opts[key] = value
    response_xml = salt.utils.namecheap.post_request(opts)
    if response_xml is None:
        return {}
    sslresult = response_xml.getElementsByTagName(result_tag_name)[0]
    result = salt.utils.namecheap.atts_to_dict(sslresult)
    if http_dc_validation:
        validation_tag = sslresult.getElementsByTagName('HttpDCValidation')
        if validation_tag:
            validation_tag = validation_tag[0]
            if validation_tag.getAttribute('ValueAvailable').lower() == 'true':
                validation_dict = {'filename': validation_tag.getElementsByTagName('FileName')[0].childNodes[0].data, 'filecontent': validation_tag.getElementsByTagName('FileContent')[0].childNodes[0].data}
                result['httpdcvalidation'] = validation_dict
    return result

def renew(years, certificate_id, certificate_type, promotion_code=None):
    if False:
        while True:
            i = 10
    "\n    Renews an SSL certificate if it is ACTIVE and Expires <= 30 days. Returns\n    the following information:\n\n    - The certificate ID\n    - The order ID\n    - The transaction ID\n    - The amount charged for the order\n\n    years : 1\n        Number of years to register\n\n    certificate_id\n        Unique ID of the SSL certificate you wish to renew\n\n    certificate_type\n        Type of SSL Certificate. Possible values include:\n\n        - EV Multi Domain SSL\n        - EV SSL\n        - EV SSL SGC\n        - EssentialSSL\n        - EssentialSSL Wildcard\n        - InstantSSL\n        - InstantSSL Pro\n        - Multi Domain SSL\n        - PositiveSSL\n        - PositiveSSL Multi Domain\n        - PositiveSSL Wildcard\n        - PremiumSSL\n        - PremiumSSL Wildcard\n        - QuickSSL Premium\n        - RapidSSL\n        - RapidSSL Wildcard\n        - SGC Supercert\n        - SSL Web Server\n        - SSL Webserver EV\n        - SSL123\n        - Secure Site\n        - Secure Site Pro\n        - Secure Site Pro with EV\n        - Secure Site with EV\n        - True BusinessID\n        - True BusinessID Multi Domain\n        - True BusinessID Wildcard\n        - True BusinessID with EV\n        - True BusinessID with EV Multi Domain\n        - Unified Communications\n\n    promotional_code\n        An optional promo code to use when renewing the certificate\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' namecheap_ssl.renew 1 my-cert-id RapidSSL\n    "
    valid_certs = ('QuickSSL Premium', 'RapidSSL', 'RapidSSL Wildcard', 'PremiumSSL', 'InstantSSL', 'PositiveSSL', 'PositiveSSL Wildcard', 'True BusinessID with EV', 'True BusinessID', 'True BusinessID Wildcard', 'True BusinessID Multi Domain', 'True BusinessID with EV Multi Domain', 'Secure Site', 'Secure Site Pro', 'Secure Site with EV', 'Secure Site Pro with EV', 'EssentialSSL', 'EssentialSSL Wildcard', 'InstantSSL Pro', 'PremiumSSL Wildcard', 'EV SSL', 'EV SSL SGC', 'SSL123', 'SSL Web Server', 'SGC Supercert', 'SSL Webserver EV', 'EV Multi Domain SSL', 'Multi Domain SSL', 'PositiveSSL Multi Domain', 'Unified Communications')
    if certificate_type not in valid_certs:
        log.error('Invalid option for certificate_type=%s', certificate_type)
        raise Exception('Invalid option for certificate_type=' + certificate_type)
    if years < 1 or years > 5:
        log.error('Invalid option for years=%s', str(years))
        raise Exception('Invalid option for years=' + str(years))
    opts = salt.utils.namecheap.get_opts('namecheap.ssl.renew')
    opts['Years'] = str(years)
    opts['CertificateID'] = str(certificate_id)
    opts['SSLType'] = certificate_type
    if promotion_code is not None:
        opts['PromotionCode'] = promotion_code
    response_xml = salt.utils.namecheap.post_request(opts)
    if response_xml is None:
        return {}
    sslrenewresult = response_xml.getElementsByTagName('SSLRenewResult')[0]
    return salt.utils.namecheap.atts_to_dict(sslrenewresult)

def create(years, certificate_type, promotion_code=None, sans_to_add=None):
    if False:
        return 10
    "\n    Creates a new SSL certificate. Returns the following information:\n\n    - Whether or not the SSL order was successful\n    - The certificate ID\n    - The order ID\n    - The transaction ID\n    - The amount charged for the order\n    - The date on which the certificate was created\n    - The date on which the certificate will expire\n    - The type of SSL certificate\n    - The number of years for which the certificate was purchased\n    - The current status of the SSL certificate\n\n    years : 1\n        Number of years to register\n\n    certificate_type\n        Type of SSL Certificate. Possible values include:\n\n        - EV Multi Domain SSL\n        - EV SSL\n        - EV SSL SGC\n        - EssentialSSL\n        - EssentialSSL Wildcard\n        - InstantSSL\n        - InstantSSL Pro\n        - Multi Domain SSL\n        - PositiveSSL\n        - PositiveSSL Multi Domain\n        - PositiveSSL Wildcard\n        - PremiumSSL\n        - PremiumSSL Wildcard\n        - QuickSSL Premium\n        - RapidSSL\n        - RapidSSL Wildcard\n        - SGC Supercert\n        - SSL Web Server\n        - SSL Webserver EV\n        - SSL123\n        - Secure Site\n        - Secure Site Pro\n        - Secure Site Pro with EV\n        - Secure Site with EV\n        - True BusinessID\n        - True BusinessID Multi Domain\n        - True BusinessID Wildcard\n        - True BusinessID with EV\n        - True BusinessID with EV Multi Domain\n        - Unified Communications\n\n    promotional_code\n        An optional promo code to use when creating the certificate\n\n    sans_to_add : 0\n        This parameter defines the number of add-on domains to be purchased in\n        addition to the default number of domains included with a multi-domain\n        certificate. Each certificate that supports SANs has the default number\n        of domains included. You may check the default number of domains\n        included and the maximum number of domains that can be added to it in\n        the table below.\n\n    +----------+----------------+----------------------+-------------------+----------------+\n    | Provider | Product name   | Default number of    | Maximum number of | Maximum number |\n    |          |                | domains (domain from | total domains     | of domains     |\n    |          |                | CSR is counted here) |                   | that can be    |\n    |          |                |                      |                   | passed in      |\n    |          |                |                      |                   | sans_to_add    |\n    |          |                |                      |                   | parameter      |\n    +----------+----------------+----------------------+-------------------+----------------+\n    | Comodo   | PositiveSSL    | 3                    | 100               | 97             |\n    |          | Multi-Domain   |                      |                   |                |\n    +----------+----------------+----------------------+-------------------+----------------+\n    | Comodo   | Multi-Domain   | 3                    | 100               | 97             |\n    |          | SSL            |                      |                   |                |\n    +----------+----------------+----------------------+-------------------+----------------+\n    | Comodo   | EV Multi-      | 3                    | 100               | 97             |\n    |          | Domain SSL     |                      |                   |                |\n    +----------+----------------+----------------------+-------------------+----------------+\n    | Comodo   | Unified        | 3                    | 100               | 97             |\n    |          | Communications |                      |                   |                |\n    +----------+----------------+----------------------+-------------------+----------------+\n    | GeoTrust | QuickSSL       | 1                    | 1 domain +        | The only       |\n    |          | Premium        |                      | 4 subdomains      | supported      |\n    |          |                |                      |                   | value is 4     |\n    +----------+----------------+----------------------+-------------------+----------------+\n    | GeoTrust | True           | 5                    | 25                | 20             |\n    |          | BusinessID     |                      |                   |                |\n    |          | with EV        |                      |                   |                |\n    |          | Multi-Domain   |                      |                   |                |\n    +----------+----------------+----------------------+-------------------+----------------+\n    | GeoTrust | True Business  | 5                    | 25                | 20             |\n    |          | ID Multi-      |                      |                   |                |\n    |          | Domain         |                      |                   |                |\n    +----------+----------------+----------------------+-------------------+----------------+\n    | Thawte   | SSL Web        | 1                    | 25                | 24             |\n    |          | Server         |                      |                   |                |\n    +----------+----------------+----------------------+-------------------+----------------+\n    | Thawte   | SSL Web        | 1                    | 25                | 24             |\n    |          | Server with    |                      |                   |                |\n    |          | EV             |                      |                   |                |\n    +----------+----------------+----------------------+-------------------+----------------+\n    | Thawte   | SGC Supercerts | 1                    | 25                | 24             |\n    +----------+----------------+----------------------+-------------------+----------------+\n    | Symantec | Secure Site    | 1                    | 25                | 24             |\n    |          | Pro with EV    |                      |                   |                |\n    +----------+----------------+----------------------+-------------------+----------------+\n    | Symantec | Secure Site    | 1                    | 25                | 24             |\n    |          | with EV        |                      |                   |                |\n    +----------+----------------+----------------------+-------------------+----------------+\n    | Symantec | Secure Site    | 1                    | 25                | 24             |\n    +----------+----------------+----------------------+-------------------+----------------+\n    | Symantec | Secure Site    | 1                    | 25                | 24             |\n    |          | Pro            |                      |                   |                |\n    +----------+----------------+----------------------+-------------------+----------------+\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' namecheap_ssl.create 2 RapidSSL\n    "
    valid_certs = ('QuickSSL Premium', 'RapidSSL', 'RapidSSL Wildcard', 'PremiumSSL', 'InstantSSL', 'PositiveSSL', 'PositiveSSL Wildcard', 'True BusinessID with EV', 'True BusinessID', 'True BusinessID Wildcard', 'True BusinessID Multi Domain', 'True BusinessID with EV Multi Domain', 'Secure Site', 'Secure Site Pro', 'Secure Site with EV', 'Secure Site Pro with EV', 'EssentialSSL', 'EssentialSSL Wildcard', 'InstantSSL Pro', 'PremiumSSL Wildcard', 'EV SSL', 'EV SSL SGC', 'SSL123', 'SSL Web Server', 'SGC Supercert', 'SSL Webserver EV', 'EV Multi Domain SSL', 'Multi Domain SSL', 'PositiveSSL Multi Domain', 'Unified Communications')
    if certificate_type not in valid_certs:
        log.error('Invalid option for certificate_type=%s', certificate_type)
        raise Exception('Invalid option for certificate_type=' + certificate_type)
    if years < 1 or years > 5:
        log.error('Invalid option for years=%s', str(years))
        raise Exception('Invalid option for years=' + str(years))
    opts = salt.utils.namecheap.get_opts('namecheap.ssl.create')
    opts['Years'] = years
    opts['Type'] = certificate_type
    if promotion_code is not None:
        opts['PromotionCode'] = promotion_code
    if sans_to_add is not None:
        opts['SANStoADD'] = sans_to_add
    response_xml = salt.utils.namecheap.post_request(opts)
    if response_xml is None:
        return {}
    sslcreateresult = response_xml.getElementsByTagName('SSLCreateResult')[0]
    sslcertinfo = sslcreateresult.getElementsByTagName('SSLCertificate')[0]
    result = salt.utils.namecheap.atts_to_dict(sslcreateresult)
    result.update(salt.utils.namecheap.atts_to_dict(sslcertinfo))
    return result

def parse_csr(csr_file, certificate_type, http_dc_validation=False):
    if False:
        i = 10
        return i + 15
    "\n    Parses the CSR. Returns a dictionary of result values.\n\n    csr_file\n        Path to Certificate Signing Request file\n\n    certificate_type\n        Type of SSL Certificate. Possible values include:\n\n        - EV Multi Domain SSL\n        - EV SSL\n        - EV SSL SGC\n        - EssentialSSL\n        - EssentialSSL Wildcard\n        - InstantSSL\n        - InstantSSL Pro\n        - Multi Domain SSL\n        - PositiveSSL\n        - PositiveSSL Multi Domain\n        - PositiveSSL Wildcard\n        - PremiumSSL\n        - PremiumSSL Wildcard\n        - QuickSSL Premium\n        - RapidSSL\n        - RapidSSL Wildcard\n        - SGC Supercert\n        - SSL Web Server\n        - SSL Webserver EV\n        - SSL123\n        - Secure Site\n        - Secure Site Pro\n        - Secure Site Pro with EV\n        - Secure Site with EV\n        - True BusinessID\n        - True BusinessID Multi Domain\n        - True BusinessID Wildcard\n        - True BusinessID with EV\n        - True BusinessID with EV Multi Domain\n        - Unified Communications\n\n    http_dc_validation : False\n        Set to ``True`` if a Comodo certificate and validation should be\n        done with files instead of emails and to return the info to do so\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' namecheap_ssl.parse_csr my-csr-file PremiumSSL\n    "
    valid_certs = ('QuickSSL Premium', 'RapidSSL', 'RapidSSL Wildcard', 'PremiumSSL', 'InstantSSL', 'PositiveSSL', 'PositiveSSL Wildcard', 'True BusinessID with EV', 'True BusinessID', 'True BusinessID Wildcard', 'True BusinessID Multi Domain', 'True BusinessID with EV Multi Domain', 'Secure Site', 'Secure Site Pro', 'Secure Site with EV', 'Secure Site Pro with EV', 'EssentialSSL', 'EssentialSSL Wildcard', 'InstantSSL Pro', 'PremiumSSL Wildcard', 'EV SSL', 'EV SSL SGC', 'SSL123', 'SSL Web Server', 'SGC Supercert', 'SSL Webserver EV', 'EV Multi Domain SSL', 'Multi Domain SSL', 'PositiveSSL Multi Domain', 'Unified Communications')
    if certificate_type not in valid_certs:
        log.error('Invalid option for certificate_type=%s', certificate_type)
        raise Exception('Invalid option for certificate_type=' + certificate_type)
    opts = salt.utils.namecheap.get_opts('namecheap.ssl.parseCSR')
    with salt.utils.files.fopen(csr_file, 'rb') as csr_handle:
        opts['csr'] = salt.utils.stringutils.to_unicode(csr_handle.read())
    opts['CertificateType'] = certificate_type
    if http_dc_validation:
        opts['HTTPDCValidation'] = 'true'
    response_xml = salt.utils.namecheap.post_request(opts)
    sslparseresult = response_xml.getElementsByTagName('SSLParseCSRResult')[0]
    return salt.utils.namecheap.xml_to_dict(sslparseresult)

def get_list(**kwargs):
    if False:
        while True:
            i = 10
    "\n    Returns a list of SSL certificates for a particular user\n\n    ListType : All\n        Possible values:\n\n        - All\n        - Processing\n        - EmailSent\n        - TechnicalProblem\n        - InProgress\n        - Completed\n        - Deactivated\n        - Active\n        - Cancelled\n        - NewPurchase\n        - NewRenewal\n\n        SearchTerm\n            Keyword to look for on the SSL list\n\n        Page : 1\n            Page number to return\n\n        PageSize : 20\n            Total number of SSL certificates to display per page (minimum:\n            ``10``, maximum: ``100``)\n\n        SoryBy\n            One of ``PURCHASEDATE``, ``PURCHASEDATE_DESC``, ``SSLTYPE``,\n            ``SSLTYPE_DESC``, ``EXPIREDATETIME``, ``EXPIREDATETIME_DESC``,\n            ``Host_Name``, or ``Host_Name_DESC``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' namecheap_ssl.get_list Processing\n    "
    opts = salt.utils.namecheap.get_opts('namecheap.ssl.getList')
    for (key, value) in kwargs.items():
        opts[key] = value
    response_xml = salt.utils.namecheap.get_request(opts)
    if response_xml is None:
        return []
    ssllistresult = response_xml.getElementsByTagName('SSLListResult')[0]
    result = []
    for e in ssllistresult.getElementsByTagName('SSL'):
        ssl = salt.utils.namecheap.atts_to_dict(e)
        result.append(ssl)
    return result

def get_info(certificate_id, returncertificate=False, returntype=None):
    if False:
        i = 10
        return i + 15
    '\n    Retrieves information about the requested SSL certificate. Returns a\n    dictionary of information about the SSL certificate with two keys:\n\n    - **ssl** - Contains the metadata information\n    - **certificate** - Contains the details for the certificate such as the\n      CSR, Approver, and certificate data\n\n    certificate_id\n        Unique ID of the SSL certificate\n\n    returncertificate : False\n        Set to ``True`` to ask for the certificate in response\n\n    returntype\n        Optional type for the returned certificate. Can be either "Individual"\n        (for X.509 format) or "PKCS7"\n\n        .. note::\n            Required if ``returncertificate`` is ``True``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'my-minion\' namecheap_ssl.get_info my-cert-id\n    '
    opts = salt.utils.namecheap.get_opts('namecheap.ssl.getinfo')
    opts['certificateID'] = certificate_id
    if returncertificate:
        opts['returncertificate'] = 'true'
        if returntype is None:
            log.error('returntype must be specified when returncertificate is set to True')
            raise Exception('returntype must be specified when returncertificate is set to True')
        if returntype not in ['Individual', 'PKCS7']:
            log.error('returntype must be specified as Individual or PKCS7, not %s', returntype)
            raise Exception('returntype must be specified as Individual or PKCS7, not ' + returntype)
        opts['returntype'] = returntype
    response_xml = salt.utils.namecheap.get_request(opts)
    if response_xml is None:
        return {}
    sslinforesult = response_xml.getElementsByTagName('SSLGetInfoResult')[0]
    return salt.utils.namecheap.xml_to_dict(sslinforesult)