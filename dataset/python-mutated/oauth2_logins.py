import json
import frappe
import frappe.utils
from frappe.utils.oauth import login_via_oauth2, login_via_oauth2_id_token

@frappe.whitelist(allow_guest=True)
def login_via_google(code: str, state: str):
    if False:
        print('Hello World!')
    login_via_oauth2('google', code, state, decoder=decoder_compat)

@frappe.whitelist(allow_guest=True)
def login_via_github(code: str, state: str):
    if False:
        while True:
            i = 10
    login_via_oauth2('github', code, state)

@frappe.whitelist(allow_guest=True)
def login_via_facebook(code: str, state: str):
    if False:
        for i in range(10):
            print('nop')
    login_via_oauth2('facebook', code, state, decoder=decoder_compat)

@frappe.whitelist(allow_guest=True)
def login_via_frappe(code: str, state: str):
    if False:
        print('Hello World!')
    login_via_oauth2('frappe', code, state, decoder=decoder_compat)

@frappe.whitelist(allow_guest=True)
def login_via_office365(code: str, state: str):
    if False:
        while True:
            i = 10
    login_via_oauth2_id_token('office_365', code, state, decoder=decoder_compat)

@frappe.whitelist(allow_guest=True)
def login_via_salesforce(code: str, state: str):
    if False:
        return 10
    login_via_oauth2('salesforce', code, state, decoder=decoder_compat)

@frappe.whitelist(allow_guest=True)
def login_via_fairlogin(code: str, state: str):
    if False:
        i = 10
        return i + 15
    login_via_oauth2('fairlogin', code, state, decoder=decoder_compat)

@frappe.whitelist(allow_guest=True)
def custom(code: str, state: str):
    if False:
        while True:
            i = 10
    '\n\tCallback for processing code and state for user added providers\n\n\tprocess social login from /api/method/frappe.integrations.oauth2_logins.custom/<provider>\n\t'
    path = frappe.request.path[1:].split('/')
    if len(path) == 4 and path[3]:
        provider = path[3]
        if frappe.db.exists('Social Login Key', provider):
            login_via_oauth2(provider, code, state, decoder=decoder_compat)

def decoder_compat(b):
    if False:
        print('Hello World!')
    return json.loads(bytes(b).decode('utf-8'))