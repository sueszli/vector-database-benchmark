import base64
import hashlib
import hmac
import re
from pyramid.view import view_config
from pyramid_jinja2 import IJinja2Environment
from warehouse.accounts.models import Email

def validate_helpscout_signature(request):
    if False:
        while True:
            i = 10
    signature = request.headers.get('X-HelpScout-Signature')
    secret = request.registry.settings.get('admin.helpscout.app_secret')
    if secret is None or signature is None:
        return False
    digest = hmac.digest(secret.encode(), request.body, hashlib.sha1)
    return hmac.compare_digest(digest, base64.b64decode(signature))

@view_config(route_name='admin.helpscout', renderer='json', require_methods=['POST'], require_csrf=False, uses_session=False)
def helpscout(request):
    if False:
        i = 10
        return i + 15
    if not validate_helpscout_signature(request):
        request.response.status = 403
        return {'Error': 'NotAuthorized'}
    email = request.db.query(Email).where(Email.email.ilike(re.sub('\\+[^)]*@', '@', request.json_body.get('customer', {}).get('email', '')))).all()
    if len(email) == 0:
        return {'html': '<span class="badge pending">No PyPI user found</span>'}
    env = request.registry.queryUtility(IJinja2Environment, name='.jinja2')
    context = {'users': [e.user for e in email]}
    template = env.get_template('admin/templates/admin/helpscout/app.html')
    content = template.render(**context, request=request)
    return {'html': content}