"""
This script submits a single MSIX installer to the Microsoft Store.

The client_secret will expire after 24 months and needs to be recreated (see docstring below).

References:
    - https://docs.microsoft.com/en-us/windows/uwp/monetize/manage-app-submissions
    - https://docs.microsoft.com/en-us/windows/uwp/monetize/python-code-examples-for-the-windows-store-submission-api
    - https://docs.microsoft.com/en-us/windows/uwp/monetize/python-code-examples-for-submissions-game-options-and-trailers
"""
import http.client
import json
import os
import sys
import tempfile
import urllib.parse
from zipfile import ZipFile
assert os.environ['GITHUB_REF'].startswith('refs/tags/') or os.environ['GITHUB_REF'] == 'refs/heads/citest'
app_id = os.environ['MSFT_APP_ID']
'\nThe public application ID / product ID of the app.\nFor https://www.microsoft.com/store/productId/9NWNDLQMNZD7, the app id is 9NWNDLQMNZD7.\n'
app_flight = os.environ.get('MSFT_APP_FLIGHT', '')
'\nThe application flight we want to target. This is useful to deploy ci test builds to a subset of users.\n'
tenant_id = os.environ['MSFT_TENANT_ID']
'\nThe tenant ID for the Azure AD application.\nhttps://partner.microsoft.com/en-us/dashboard/account/v3/usermanagement\n'
client_id = os.environ['MSFT_CLIENT_ID']
'\nThe client ID for the Azure AD application.\nhttps://partner.microsoft.com/en-us/dashboard/account/v3/usermanagement\n'
client_secret = os.environ['MSFT_CLIENT_SECRET']
'\nThe client secret. Expires every 24 months and needs to be recreated at\nhttps://partner.microsoft.com/en-us/dashboard/account/v3/usermanagement\nor at https://portal.azure.com/ -> App registrations -> Certificates & Secrets -> Client secrets.\n'
try:
    (_, msi_file) = sys.argv
except ValueError:
    print(f'Usage: {sys.argv[0]} installer.msix')
    sys.exit(1)
if app_flight:
    app_id = f'{app_id}/flights/{app_flight}'
    pending_submission = 'pendingFlightSubmission'
    packages = 'flightPackages'
else:
    pending_submission = 'pendingApplicationSubmission'
    packages = 'applicationPackages'
print('Obtaining auth token...')
auth = http.client.HTTPSConnection('login.microsoftonline.com')
auth.request('POST', f'/{tenant_id}/oauth2/token', body=urllib.parse.urlencode({'grant_type': 'client_credentials', 'client_id': client_id, 'client_secret': client_secret, 'resource': 'https://manage.devcenter.microsoft.com'}), headers={'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8'})
token = json.loads(auth.getresponse().read())['access_token']
auth.close()
headers = {'Authorization': f'Bearer {token}', 'Content-type': 'application/json', 'User-Agent': 'Python/mitmproxy'}

def request(method: str, path: str, body: str='') -> bytes:
    if False:
        i = 10
        return i + 15
    print(f'{method} {path}')
    conn.request(method, path, body, headers=headers)
    resp = conn.getresponse()
    data = resp.read()
    print(f'{resp.status} {resp.reason}')
    if False:
        assert 'CI' not in os.environ
        print(data.decode(errors='ignore'))
    assert 200 <= resp.status < 300
    return data
print('Getting app info...')
conn = http.client.HTTPSConnection('manage.devcenter.microsoft.com')
app_info = json.loads(request('GET', f'/v1.0/my/applications/{app_id}'))
if pending_submission in app_info:
    print('Deleting pending submission...')
    request('DELETE', f"/v1.0/my/applications/{app_id}/submissions/{app_info[pending_submission]['id']}")
print('Creating new submission...')
submission = json.loads(request('POST', f'/v1.0/my/applications/{app_id}/submissions'))
print('Updating submission...')
for package in submission[packages]:
    package['fileStatus'] = 'PendingDelete'
submission[packages].append({'fileName': f'installer.msix', 'fileStatus': 'PendingUpload', 'minimumDirectXVersion': 'None', 'minimumSystemRam': 'None'})
request('PUT', f"/v1.0/my/applications/{app_id}/submissions/{submission['id']}", json.dumps(submission))
conn.close()
print(f'Zipping {msi_file}...')
with tempfile.TemporaryFile() as zipfile:
    with ZipFile(zipfile, 'w') as f:
        f.write(msi_file, f'installer.msix')
    zip_size = zipfile.tell()
    zipfile.seek(0)
    print('Uploading zip file...')
    (host, _, path) = submission['fileUploadUrl'].removeprefix('https://').partition('/')
    upload = http.client.HTTPSConnection(host)
    upload.request('PUT', '/' + path, zipfile, {'x-ms-blob-type': 'BlockBlob', 'x-ms-version': '2019-12-12', 'Content-Length': str(zip_size)})
resp = upload.getresponse()
resp.read()
print(resp.status, resp.reason)
assert 200 <= resp.status < 300
upload.close()
print('Publishing submission...')
conn = http.client.HTTPSConnection('manage.devcenter.microsoft.com')
request('POST', f"/v1.0/my/applications/{app_id}/submissions/{submission['id']}/commit")
conn.close()