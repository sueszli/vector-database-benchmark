"""Example ACME-V2 API for HTTP-01 challenge.

Brief:

This a complete usage example of the python-acme API.

Limitations of this example:
    - Works for only one Domain name
    - Performs only HTTP-01 challenge
    - Uses ACME-v2

Workflow:
    (Account creation)
    - Create account key
    - Register account and accept TOS
    (Certificate actions)
    - Select HTTP-01 within offered challenges by the CA server
    - Set up http challenge resource
    - Set up standalone web server
    - Create domain private key and CSR
    - Issue certificate
    - Renew certificate
    - Revoke certificate
    (Account update actions)
    - Change contact information
    - Deactivate Account
"""
from contextlib import contextmanager
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
import josepy as jose
import OpenSSL
from acme import challenges
from acme import client
from acme import crypto_util
from acme import errors
from acme import messages
from acme import standalone
DIRECTORY_URL = 'https://acme-staging-v02.api.letsencrypt.org/directory'
USER_AGENT = 'python-acme-example'
ACC_KEY_BITS = 2048
CERT_PKEY_BITS = 2048
DOMAIN = 'client.example.com'
PORT = 80

def new_csr_comp(domain_name, pkey_pem=None):
    if False:
        return 10
    'Create certificate signing request.'
    if pkey_pem is None:
        pkey = OpenSSL.crypto.PKey()
        pkey.generate_key(OpenSSL.crypto.TYPE_RSA, CERT_PKEY_BITS)
        pkey_pem = OpenSSL.crypto.dump_privatekey(OpenSSL.crypto.FILETYPE_PEM, pkey)
    csr_pem = crypto_util.make_csr(pkey_pem, [domain_name])
    return (pkey_pem, csr_pem)

def select_http01_chall(orderr):
    if False:
        for i in range(10):
            print('nop')
    'Extract authorization resource from within order resource.'
    authz_list = orderr.authorizations
    for authz in authz_list:
        for i in authz.body.challenges:
            if isinstance(i.chall, challenges.HTTP01):
                return i
    raise Exception('HTTP-01 challenge was not offered by the CA server.')

@contextmanager
def challenge_server(http_01_resources):
    if False:
        print('Hello World!')
    'Manage standalone server set up and shutdown.'
    address = ('', PORT)
    try:
        servers = standalone.HTTP01DualNetworkedServers(address, http_01_resources)
        servers.serve_forever()
        yield servers
    finally:
        servers.shutdown_and_server_close()

def perform_http01(client_acme, challb, orderr):
    if False:
        for i in range(10):
            print('nop')
    'Set up standalone webserver and perform HTTP-01 challenge.'
    (response, validation) = challb.response_and_validation(client_acme.net.key)
    resource = standalone.HTTP01RequestHandler.HTTP01Resource(chall=challb.chall, response=response, validation=validation)
    with challenge_server({resource}):
        client_acme.answer_challenge(challb, response)
        finalized_orderr = client_acme.poll_and_finalize(orderr)
    return finalized_orderr.fullchain_pem

def example_http():
    if False:
        i = 10
        return i + 15
    'This example executes the whole process of fulfilling a HTTP-01\n    challenge for one specific domain.\n\n    The workflow consists of:\n    (Account creation)\n    - Create account key\n    - Register account and accept TOS\n    (Certificate actions)\n    - Select HTTP-01 within offered challenges by the CA server\n    - Set up http challenge resource\n    - Set up standalone web server\n    - Create domain private key and CSR\n    - Issue certificate\n    - Renew certificate\n    - Revoke certificate\n    (Account update actions)\n    - Change contact information\n    - Deactivate Account\n\n    '
    acc_key = jose.JWKRSA(key=rsa.generate_private_key(public_exponent=65537, key_size=ACC_KEY_BITS, backend=default_backend()))
    net = client.ClientNetwork(acc_key, user_agent=USER_AGENT)
    directory = client.ClientV2.get_directory(DIRECTORY_URL, net)
    client_acme = client.ClientV2(directory, net=net)
    email = 'fake@example.com'
    regr = client_acme.new_account(messages.NewRegistration.from_data(email=email, terms_of_service_agreed=True))
    (pkey_pem, csr_pem) = new_csr_comp(DOMAIN)
    orderr = client_acme.new_order(csr_pem)
    challb = select_http01_chall(orderr)
    fullchain_pem = perform_http01(client_acme, challb, orderr)
    (_, csr_pem) = new_csr_comp(DOMAIN, pkey_pem)
    orderr = client_acme.new_order(csr_pem)
    challb = select_http01_chall(orderr)
    fullchain_pem = perform_http01(client_acme, challb, orderr)
    fullchain_com = jose.ComparableX509(OpenSSL.crypto.load_certificate(OpenSSL.crypto.FILETYPE_PEM, fullchain_pem))
    try:
        client_acme.revoke(fullchain_com, 0)
    except errors.ConflictError:
        pass
    client_acme.net.account = regr
    try:
        regr = client_acme.query_registration(regr)
    except errors.Error as err:
        if err.typ == messages.ERROR_PREFIX + 'unauthorized':
            pass
        raise
    email = 'newfake@example.com'
    regr = client_acme.update_registration(regr.update(body=regr.body.update(contact=('mailto:' + email,))))
    regr = client_acme.deactivate_registration(regr)
if __name__ == '__main__':
    example_http()