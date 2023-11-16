from os.path import abspath, dirname, join as pjoin
import pprint
import sys
import time
from urllib.parse import urlparse
import requests
from icecream import ic
_corePath = abspath(pjoin(dirname(__file__), '../'))
if _corePath not in sys.path:
    sys.path.append(_corePath)
from common.utils import lget, stripStringStart
DEFAULT_FIRST_NAME = 'there'
DEFAULT_LAST_NAME = '-'
FRESH_SALES_API_KEY = 'P3bYheaquAHH1_hNxhMUDQ'
FS_API_URL = 'https://arcindustriesinc.freshsales.io/api'
FS_AUTH_HEADERS = {'Authorization': 'Token token=P3bYheaquAHH1_hNxhMUDQ'}

def splitName(name):
    if False:
        i = 10
        return i + 15
    toks = name.split(None, 1)
    firstName = lget(toks, 0, DEFAULT_FIRST_NAME)
    lastName = lget(toks, 1, DEFAULT_LAST_NAME)
    return (firstName, lastName)

def lookupFullContact(contact):
    if False:
        for i in range(10):
            print('nop')
    contactId = contact['id']
    resp = requests.get(f'{FS_API_URL}/contacts/{contactId}?include=sales_accounts', headers=FS_AUTH_HEADERS)
    contact = (resp.json() or {}).get('contact')
    return contact

def findFirstContactWithEmail(emailAddr):
    if False:
        while True:
            i = 10
    contacts = _findEntitiesWith('contact', 'email', emailAddr)
    return lget(contacts, 0)

def findFirstCompanyWithWebsite(websiteUrl):
    if False:
        while True:
            i = 10
    hostNoWww = lambda url: stripStringStart(urlparse(url).hostname, 'www.')
    allCompanies = _findEntitiesWith('sales_account', 'website', websiteUrl)
    ic(allCompanies)
    companies = [c for c in _findEntitiesWith('sales_account', 'website', websiteUrl) if ic(hostNoWww(c.get('website'))) == ic(hostNoWww(websiteUrl))]
    firstCompany = lget(companies, 0)
    return firstCompany

def _findEntitiesWith(entityType, query, queryValue):
    if False:
        while True:
            i = 10
    resp = requests.get(f'{FS_API_URL}/lookup?f={query}&entities={entityType}', params={'q': queryValue}, headers=FS_AUTH_HEADERS)
    entities = (resp.json() or {}).get(f'{entityType}s', {}).get(f'{entityType}s', [])
    return entities

def createNote(entityType, entityId, message):
    if False:
        print('Hello World!')
    data = {'note': {'description': message, 'targetable_id': entityId, 'targetable_type': entityType}}
    resp = requests.post(f'{FS_API_URL}/notes', json=data, headers=FS_AUTH_HEADERS)
    if resp.status_code != 201:
        err = f'Failed to create {entityType} note for id {entityId}.'
        raise RuntimeError(err)

def createLead(data):
    if False:
        return 10
    return _createEntity('lead', data)

def createContact(data):
    if False:
        for i in range(10):
            print('nop')
    ANSGAR_GRUNSEID = 9000013180
    data.setdefault('owner_id', ANSGAR_GRUNSEID)
    return _createEntity('contact', data)

def createCompany(data):
    if False:
        while True:
            i = 10
    return _createEntity('sales_account', data)

def _createEntity(entityType, data):
    if False:
        print('Hello World!')
    wrapped = {entityType: data}
    url = f'{FS_API_URL}/{entityType}s'
    resp = requests.post(url, json=wrapped, headers=FS_AUTH_HEADERS)
    if resp.status_code not in [200, 201]:
        raise RuntimeError(f'Failed to create new {entityType}.')
    entity = (resp.json() or {}).get(entityType)
    return entity

def updateLead(leadId, data):
    if False:
        for i in range(10):
            print('nop')
    return _updateEntity('lead', leadId, data)

def updateContact(contactId, data):
    if False:
        print('Hello World!')
    return _updateEntity('contact', contactId, data)

def updateCompany(companyId, data):
    if False:
        for i in range(10):
            print('nop')
    return _updateEntity('sales_account', companyId, data)

def _updateEntity(entityType, entityId, data):
    if False:
        i = 10
        return i + 15
    wrapped = {entityType: data}
    url = f'{FS_API_URL}/{entityType.lower()}s/{entityId}'
    resp = requests.put(url, json=wrapped, headers=FS_AUTH_HEADERS)
    if resp.status_code != 200:
        err = f'Failed to update {entityType.title()} with id {entityId}.'
        raise RuntimeError(err)
    entity = (resp.json() or {}).get(entityType)
    return entity

def lookupContactsInView(viewId):
    if False:
        return 10
    return _lookupEntitiesInView('contact', viewId)

def _lookupEntitiesInView(entityType, viewId):
    if False:
        i = 10
        return i + 15
    entities = []
    url = f'{FS_API_URL}/{entityType.lower()}s/view/{viewId}'

    def pageUrl(pageNo):
        if False:
            while True:
                i = 10
        return url + f'?page={pageNo}'
    resp = requests.get(url, headers=FS_AUTH_HEADERS)
    js = resp.json()
    entities += js.get(f'{entityType}s')
    totalPages = js.get('meta', {}).get('total_pages')
    for pageNo in range(2, totalPages + 1):
        resp = requests.get(pageUrl(pageNo), headers=FS_AUTH_HEADERS)
        entities += (resp.json() or {}).get(f'{entityType}s')
    return entities

def unsubscribeContact(contact, reasons):
    if False:
        i = 10
        return i + 15
    UNSUBSCRIBED = 9000159966
    updateContact(contact['id'], {'do_not_disturb': True, 'contact_status_id': UNSUBSCRIBED})
    dateStr = time.ctime()
    reasonsStr = pprint.pformat(reasons)
    note = f'This Contact unsubscribed on arc.io/unsubscribe at [{dateStr}] because:\n\n{reasonsStr}\n\n'
    createNote('Contact', contact['id'], note)

def optContactIn(contact):
    if False:
        for i in range(10):
            print('nop')
    OPTED_IN = 9000159976
    updateContact(contact['id'], {'contact_status_id': OPTED_IN})

def createAndOrAssociateCompanyWithContact(websiteUrl, contact):
    if False:
        while True:
            i = 10
    if 'sales_accounts' not in contact:
        contact = lookupFullContact(contact)
    companyToAdd = None
    companies = contact.get('sales_accounts', [])
    company = findFirstCompanyWithWebsite(websiteUrl)
    if company:
        companyId = company['id']
        alreadyRelated = any((companyId == c['id'] for c in companies))
        if not alreadyRelated:
            companyToAdd = company
    else:
        companyToAdd = createCompany({'name': websiteUrl, 'website': websiteUrl})
    if companyToAdd:
        companyData = {'id': companyToAdd['id'], 'is_primary': False if companies else True}
        companies.append(companyData)
    updateContact(contact['id'], {'sales_accounts': companies})
    return company or companyToAdd

def upgradeContactWhoSubmittedSplashPage(contact, websiteUrl):
    if False:
        for i in range(10):
            print('nop')
    createAndOrAssociateCompanyWithContact(websiteUrl, contact)
    SUBMITTED_ARC_IO_SIGN_UP_FORM = 9000159955
    updateContact(contact['id'], {'contact_status_id': SUBMITTED_ARC_IO_SIGN_UP_FORM})
    dateStr = time.ctime()
    emailAddr = contact['email']
    note = f'This Contact submitted the sign up form on arc.io at [{dateStr}] with email address [{emailAddr}] and website [{websiteUrl}].'
    createNote('Contact', contact['id'], note)

def noteContactSubmittedPepSplashPage(contact, websiteUrl):
    if False:
        while True:
            i = 10
    createAndOrAssociateCompanyWithContact(websiteUrl, contact)
    PEP = 9000004543
    updateContact(contact['id'], {'custom_field': {'cf_product': 'Pep'}})
    dateStr = time.ctime()
    emailAddr = contact['email']
    note = f"This Contact submitted Pep's sign up form on pep.dev at [{dateStr}] with email address [{emailAddr}] and website [{websiteUrl}]."
    createNote('Contact', contact['id'], note)

def createCrawledIndieHackersContact(name, emailAddr, websiteUrl, noteData):
    if False:
        return 10
    INDIE_HACKERS = 9000321821
    _createCrawledContact(name, emailAddr, websiteUrl, INDIE_HACKERS, noteData)

def _createCrawledContact(name, emailAddr, websiteUrl, leadSourceId, noteData):
    if False:
        print('Hello World!')
    (firstName, lastName) = splitName(name)
    SUSPECT = 9000073090
    contact = createContact({'email': emailAddr, 'first_name': firstName, 'last_name': lastName, 'contact_status_id': SUSPECT, 'lead_source_id': leadSourceId})
    createAndOrAssociateCompanyWithContact(websiteUrl, contact)
    dateStr = time.ctime()
    reasonsStr = pprint.pformat(noteData)
    note = f'This Contact was crawled and created on [{dateStr}]. Other data:\n{reasonsStr}\n\n'
    createNote('Contact', contact['id'], note)

def createSplashPageLead(name, emailAddr, websiteUrl):
    if False:
        while True:
            i = 10
    (firstName, lastName) = splitName(name)
    INTERESTED = 9000057526
    ARC_IO_SIGN_UP_FORM = 9000315608
    lead = createLead({'first_name': firstName, 'last_name': lastName, 'email': emailAddr, 'company': {'website': websiteUrl}, 'lead_stage_id': INTERESTED, 'lead_source_id': ARC_IO_SIGN_UP_FORM})
    dateStr = time.ctime()
    note = f'This Lead was created on [{dateStr}] because they submitted the sign up form on arc.io with email address [{emailAddr}] and website [{websiteUrl}].'
    createNote('Lead', lead['id'], note)

def createPepSplashPageLead(emailAddr, websiteUrl):
    if False:
        while True:
            i = 10
    PEP = 9000004543
    INTERESTED = 9000057526
    PEP_SIGN_UP_FORM = 9000321929
    lead = createLead({'first_name': DEFAULT_FIRST_NAME, 'last_name': DEFAULT_LAST_NAME, 'email': emailAddr, 'company': {'website': websiteUrl}, 'deal': {'deal_product_id': PEP}, 'lead_stage_id': INTERESTED, 'lead_source_id': PEP_SIGN_UP_FORM})
    dateStr = time.ctime()
    note = f'This Lead was created on [{dateStr}] because they submitted the sign up form on pep.dev with email address [{emailAddr}] and website [{websiteUrl}].'
    createNote('Lead', lead['id'], note)

def noteACustomersFirstWidgetReport(emailAddr, seenOnUrl):
    if False:
        return 10
    raise NotImplementedError
    contact = findFirstContactWithEmail(emailAddr)
    if contact:
        note = f'The widget for Arc account with email {emailAddr} was just seen live for the first seen for the first time live on {seenOnUrl}.'
        createNote('Contact', contact['id'], note)
    else:
        ic()

def handleWordPressPluginInstall(emailAddr, websiteUrl):
    if False:
        print('Hello World!')
    WORDPRESS = 9000321857
    ALPHA_CODE = 9000124404
    contact = findFirstContactWithEmail(emailAddr)
    if contact:
        updateContact(contact['id'], {'lead_source_id': WORDPRESS, 'contact_status_id': ALPHA_CODE})
    else:
        contact = createContact({'email': emailAddr, 'first_name': 'there', 'last_name': websiteUrl, 'lead_source_id': WORDPRESS, 'contact_status_id': ALPHA_CODE})
    CUSTOMER = 9000095000
    company = createAndOrAssociateCompanyWithContact(websiteUrl, contact)
    updateCompany(company['id'], {'business_type_id': CUSTOMER, 'custom_field': {'cf_source': 'Wordpress'}})
    dateStr = time.ctime()
    note = f"This Contact installed Arc's WordPress plugin at [{dateStr}] on website [{{websiteUrl}}]."
    createNote('Contact', contact['id'], note)

def handleWordPressPluginCreatedArcAccount(emailAddr):
    if False:
        while True:
            i = 10
    contact = findFirstContactWithEmail(emailAddr)
    if not contact:
        return
    CUSTOMER = 9000066454
    updateContact(contact['id'], {'contact_status_id': CUSTOMER})
    dateStr = time.ctime()
    note = f'This WordPress Contact created their Arc account at [{dateStr}].'
    createNote('Contact', contact['id'], note)

def handleWordPressPluginUninstall(emailAddr):
    if False:
        for i in range(10):
            print('nop')
    contact = findFirstContactWithEmail(emailAddr)
    if not contact:
        return
    FORMER_CUSTOMER = 9000124405
    updateContact(contact['id'], {'contact_status_id': FORMER_CUSTOMER})
    dateStr = time.ctime()
    note = f'This Contact uninstalled their WordPress plugin at [{dateStr}].'
    createNote('Contact', contact['id'], note)
if __name__ == '__main__':
    ic(findFirstCompanyWithWebsite('http://blockchainexamples.com'))