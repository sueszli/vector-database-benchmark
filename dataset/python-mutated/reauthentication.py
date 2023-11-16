import time
from django.contrib.auth import REDIRECT_FIELD_NAME
from django.http import HttpResponseRedirect
from django.urls import resolve, reverse
from django.utils.http import urlencode
from allauth.account import app_settings
from allauth.account.adapter import get_adapter
from allauth.account.utils import get_next_redirect_url
from allauth.core.internal.http import deserialize_request, serialize_request
from allauth.utils import import_callable
STATE_SESSION_KEY = 'account_reauthentication_state'
AUTHENTICATED_AT_SESSION_KEY = 'account_authenticated_at'

def suspend_request(request, redirect_to):
    if False:
        for i in range(10):
            print('nop')
    path = request.get_full_path()
    if request.method == 'POST':
        request.session[STATE_SESSION_KEY] = {'request': serialize_request(request)}
    return HttpResponseRedirect(redirect_to + '?' + urlencode({REDIRECT_FIELD_NAME: path}))

def resume_request(request):
    if False:
        return 10
    state = request.session.pop(STATE_SESSION_KEY, None)
    if state and 'callback' in state:
        callback = import_callable(state['callback'])
        return callback(request, state['state'])
    url = get_next_redirect_url(request, REDIRECT_FIELD_NAME)
    if not url:
        return None
    if state and 'request' in state:
        suspended_request = deserialize_request(state['request'], request)
        if suspended_request.path == url:
            resolved = resolve(suspended_request.path)
            return resolved.func(suspended_request, *resolved.args, **resolved.kwargs)
    return HttpResponseRedirect(url)

def record_authentication(request, user):
    if False:
        for i in range(10):
            print('nop')
    request.session[AUTHENTICATED_AT_SESSION_KEY] = time.time()

def reauthenticate_then_callback(request, serialize_state, callback):
    if False:
        while True:
            i = 10
    if did_recently_authenticate(request):
        return None
    request.session[STATE_SESSION_KEY] = {'state': serialize_state(request), 'callback': callback}
    return HttpResponseRedirect(reverse('account_reauthenticate'))

def did_recently_authenticate(request):
    if False:
        i = 10
        return i + 15
    if request.user.is_anonymous:
        return False
    if not get_adapter().get_reauthentication_methods(request.user):
        return True
    authenticated_at = request.session.get(AUTHENTICATED_AT_SESSION_KEY)
    if not authenticated_at:
        return False
    return time.time() - authenticated_at < app_settings.REAUTHENTICATION_TIMEOUT