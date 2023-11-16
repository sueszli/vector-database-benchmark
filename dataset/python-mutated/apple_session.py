from allauth.socialaccount.sessions import LoginSession
APPLE_SESSION_COOKIE_NAME = 'apple-login-session'

def get_apple_session(request):
    if False:
        for i in range(10):
            print('nop')
    return LoginSession(request, 'apple_login_session', APPLE_SESSION_COOKIE_NAME)