"""
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import remi.gui as gui
from remi import start, App
import random
import threading

class CookieInterface(gui.Tag, gui.EventSource):

    def __init__(self, remi_app_instance, **kwargs):
        if False:
            print('Hello World!')
        '\n        This class uses javascript code from cookie.js framework ( https://developer.mozilla.org/en-US/docs/Web/API/document.cookie )\n        /*        |*|\n        |*|  :: cookies.js ::\n        |*|\n        |*|  A complete cookies reader/writer framework with full unicode support.\n        |*|\n        |*|  Revision #2 - June 13th, 2017\n        |*|\n        |*|  https://developer.mozilla.org/en-US/docs/Web/API/document.cookie\n        |*|  https://developer.mozilla.org/User:fusionchess\n        |*|  https://github.com/madmurphy/cookies.js\n        |*|\n        |*|  This framework is released under the GNU Public License, version 3 or later.\n        |*|  http://www.gnu.org/licenses/gpl-3.0-standalone.html\n        |*|\n        \\*/\n        '
        super(CookieInterface, self).__init__(**kwargs)
        gui.EventSource.__init__(self)
        self.app_instance = remi_app_instance
        self.EVENT_ONCOOKIES = 'on_cookies'
        self.cookies = {}

    def request_cookies(self):
        if False:
            while True:
                i = 10
        self.app_instance.execute_javascript('\n            var aKeys = document.cookie.replace(/((?:^|\\s*;)[^\\=]+)(?=;|$)|^\\s*|\\s*(?:\\=[^;]*)?(?:\x01|$)/g, "").split(/\\s*(?:\\=[^;]*)?;\\s*/);\n            var result = {};\n            for (var nLen = aKeys.length, nIdx = 0; nIdx < nLen; nIdx++) { \n                var key = decodeURIComponent(aKeys[nIdx]);\n                result[key] = decodeURIComponent(document.cookie.replace(new RegExp("(?:(?:^|.*;)\\s*" + encodeURIComponent(key).replace(/[\\-\\.\\+\\*]/g, "\\$&") + "\\s*\\=\\s*([^;]*).*$)|^.*$"), "$1")) || null; \n            }\n            remi.sendCallbackParam(\'%s\',\'%s\', result);\n            ' % (self.identifier, self.EVENT_ONCOOKIES))

    @gui.decorate_event
    def on_cookies(self, **value):
        if False:
            return 10
        self.cookies = value
        return (value,)

    def remove_cookie(self, key, path='/', domain=''):
        if False:
            while True:
                i = 10
        if not key in self.cookies.keys():
            return
        self.app_instance.execute_javascript('\n            var sKey = "%(sKey)s";\n            var sPath = "%(sPath)s";\n            var sDomain = "%(sDomain)s";\n            document.cookie = encodeURIComponent(sKey) + "=; expires=Thu, 01 Jan 1970 00:00:00 GMT" + (sDomain ? "; domain=" + sDomain : "") + (sPath ? "; path=" + sPath : "");\n            ' % {'sKey': key, 'sPath': path, 'sDomain': domain})

    def set_cookie(self, key, value, expiration='Infinity', path='/', domain='', secure=False):
        if False:
            i = 10
            return i + 15
        '\n        expiration (int): seconds after with the cookie automatically gets deleted\n        '
        secure = 'true' if secure else 'false'
        self.app_instance.execute_javascript('\n            var sKey = "%(sKey)s";\n            var sValue = "%(sValue)s";\n            var vEnd = eval("%(vEnd)s");\n            var sPath = "%(sPath)s"; \n            var sDomain = "%(sDomain)s"; \n            var bSecure = %(bSecure)s;\n            if( (!sKey || /^(?:expires|max\\-age|path|domain|secure)$/i.test(sKey)) == false ){\n                var sExpires = "";\n                if (vEnd) {\n                    switch (vEnd.constructor) {\n                        case Number:\n                            sExpires = vEnd === Infinity ? "; expires=Fri, 31 Dec 9999 23:59:59 GMT" : "; max-age=" + vEnd;\n                        break;\n                        case String:\n                            sExpires = "; expires=" + vEnd;\n                        break;\n                        case Date:\n                            sExpires = "; expires=" + vEnd.toUTCString();\n                        break;\n                    }\n                }\n                document.cookie = encodeURIComponent(sKey) + "=" + encodeURIComponent(sValue) + sExpires + (sDomain ? "; domain=" + sDomain : "") + (sPath ? "; path=" + sPath : "") + (bSecure ? "; secure" : "");\n            }\n            ' % {'sKey': key, 'sValue': value, 'vEnd': expiration, 'sPath': path, 'sDomain': domain, 'bSecure': secure})

class LoginManager(gui.Tag, gui.EventSource):
    """
    Login manager class allows to simply manage user access safety by session cookies
    It requires a cookieInterface instance to query and set user session id
    When the user login to the system you have to call
        login_manager.renew_session() #in order to force new session uid setup

    The session have to be refreshed each user action (like button click or DB access)
    in order to avoid expiration. BUT before renew, check if expired in order to ask user login

        if not login_manager.expired:
            login_manager.renew_session()
            #RENEW OK
        else:
            #UNABLE TO RENEW
            #HAVE TO ASK FOR LOGIN

    In order to know session expiration, you should register to on_session_expired event
        on_session_expired.do(mylistener.on_user_logout)
    When this event happens, ask for user login
    """

    def __init__(self, cookieInterface, session_timeout_seconds=60, **kwargs):
        if False:
            while True:
                i = 10
        super(LoginManager, self).__init__(**kwargs)
        gui.EventSource.__init__(self)
        self.expired = True
        self.session_uid = str(random.randint(1, 999999999))
        self.cookieInterface = cookieInterface
        self.session_timeout_seconds = session_timeout_seconds
        self.timer_request_cookies()
        self.timeout_timer = None

    def timer_request_cookies(self):
        if False:
            print('Hello World!')
        self.cookieInterface.request_cookies()
        self.cookie_timer = threading.Timer(self.session_timeout_seconds / 10.0, self.timer_request_cookies)
        self.cookie_timer.daemon = True
        self.cookie_timer.start()

    @gui.decorate_event
    def on_session_expired(self):
        if False:
            for i in range(10):
                print('nop')
        self.expired = True
        return ()

    def renew_session(self):
        if False:
            while True:
                i = 10
        'Have to be called on user actions to check and renew session\n        '
        if (not 'user_uid' in self.cookieInterface.cookies or self.cookieInterface.cookies['user_uid'] != self.session_uid) and (not self.expired):
            self.on_session_expired()
        if self.expired:
            self.session_uid = str(random.randint(1, 999999999))
        self.cookieInterface.set_cookie('user_uid', self.session_uid, str(self.session_timeout_seconds))
        if self.timeout_timer:
            self.timeout_timer.cancel()
        self.timeout_timer = threading.Timer(self.session_timeout_seconds, self.on_session_expired)
        self.timeout_timer.daemon = True
        self.expired = False
        self.timeout_timer.start()

class MyApp(App):

    def __init__(self, *args):
        if False:
            while True:
                i = 10
        super(MyApp, self).__init__(*args)

    def main(self, name='world'):
        if False:
            print('Hello World!')
        self.login_manager = LoginManager(CookieInterface(self), 5)
        self.login_manager.on_session_expired.do(self.on_logout)
        wid = gui.VBox(width=200, height=300, margin='0px auto')
        btlogin = gui.Button('LOGIN')
        btlogin.onclick.do(self.on_login)
        btrenew = gui.Button('RENEW BEFORE EXPIRATION')
        btrenew.onclick.do(self.on_renew)
        self.lblsession_status = gui.Label('NOT LOGGED IN')
        wid.append(btlogin)
        wid.append(btrenew)
        wid.append(self.lblsession_status)
        return wid

    def on_login(self, emitter):
        if False:
            i = 10
            return i + 15
        self.login_manager.renew_session()
        self.lblsession_status.set_text('LOGGED IN')

    def on_renew(self, emitter):
        if False:
            print('Hello World!')
        if not self.login_manager.expired:
            self.login_manager.renew_session()
            self.lblsession_status.set_text('RENEW')
        else:
            self.lblsession_status.set_text('UNABLE TO RENEW')

    def on_logout(self, emitter):
        if False:
            return 10
        self.lblsession_status.set_text('LOGOUT')
if __name__ == '__main__':
    start(MyApp, address='0.0.0.0', port=0, multiple_instance=False, debug=False)