import re
import urllib.parse
from ..base.captcha_service import CaptchaService

class HCaptcha(CaptchaService):
    __name__ = 'HCaptcha'
    __type__ = 'anticaptcha'
    __version__ = '0.03'
    __status__ = 'testing'
    __description__ = 'hCaptcha captcha service plugin'
    __license__ = 'GPLv3'
    __authors__ = [('GammaC0de', 'nitzo2001[AT]yahoo[DOT]com')]
    KEY_PATTERN = '(?:data-sitekey=["\\\']|["\\\']sitekey["\\\']\\s*:\\s*["\\\'])((?:[\\w\\-]|%[0-9a-fA-F]{2})+)'
    KEY_FORMAT_PATTERN = '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    HCAPTCHA_INTERACTIVE_SIG = '8595965f0f9fd776231ff65294f515e74384063b64c31d68bdb98ee33a500ab9a5f4ecaf84d24b77' + '621b516fc8799bbe284e5e6985dcbd6b05dbcb8a5704d75fc566ec8da380ab30d0b65afbd0300169' + '57432878020c737cee20746c350c79a8888f087041b443e6def2587d9ff6f5ca52a1476dae96c9d1' + '0d8f3b95edd337a0e8344435a563756eca3b2e4a76fb9581fe265fa8ff6040d1c64180c67798b063' + '2beb5a8b95f3bc2c16c545bd7335ebaeb9d2fcd654bef7bf51ac1907204eaf9fe1671118c4cf9011' + '45722303d475f1ab876a75062ba60d17f48da3d2bf795e09749660ce1348a9aeb228eb2129caa565' + '9268b345ee306113e0c5a2c436cbb0aa'
    HCAPTCHA_INTERACTIVE_JS = '\n\t\t\twhile(document.children[0].childElementCount > 0) {\n\t\t\t\tdocument.children[0].removeChild(document.children[0].children[0]);\n\t\t\t}\n\t\t\tdocument.children[0].innerHTML = \'<html><head></head><body style="display:inline-block;"><div id="captchadiv" style="display: inline-block;"></div></body></html>\';\n\n\t\t\tgpyload.data.sitekey = request.params.sitekey;\n\n\t\t\tgpyload.getFrameSize = function() {\n\t\t\t\tvar rectAnchor =  {top: 0, right: 0, bottom: 0, left: 0},\n\t\t\t\t\trectPopup =  {top: 0, right: 0, bottom: 0, left: 0},\n\t\t\t\t\trect;\n\t\t\t\tvar anchor = document.body.querySelector("iframe[src*=\'/hcaptcha.html#frame=checkbox\']");\n\t\t\t\tif (anchor !== null && gpyload.isVisible(anchor)) {\n\t\t\t\t\trect = anchor.getBoundingClientRect();\n\t\t\t\t\trectAnchor = {top: rect.top, right: rect.right, bottom: rect.bottom, left: rect.left};\n\t\t\t\t}\n\t\t\t\tvar popup = document.body.querySelector("iframe[src*=\'/hcaptcha.html#frame=challenge\']");\n\t\t\t\tif (popup !== null && gpyload.isVisible(popup)) {\n\t\t\t\t\trect = popup.getBoundingClientRect();\n\t\t\t\t\trectPopup = {top: rect.top, right: rect.right, bottom: rect.bottom, left: rect.left};\n\t\t\t\t}\n\t\t\t\tvar left = Math.round(Math.min(rectAnchor.left, rectAnchor.right, rectPopup.left, rectPopup.right));\n\t\t\t\tvar right = Math.round(Math.max(rectAnchor.left, rectAnchor.right, rectPopup.left, rectPopup.right));\n\t\t\t\tvar top = Math.round(Math.min(rectAnchor.top, rectAnchor.bottom, rectPopup.top, rectPopup.bottom));\n\t\t\t\tvar bottom = Math.round(Math.max(rectAnchor.top, rectAnchor.bottom, rectPopup.top, rectPopup.bottom));\n\t\t\t\treturn {top: top, left: left, bottom: bottom, right: right};\n\t\t\t};\n\n\t\t\t// function that is called when the captcha finished loading and is ready to interact\n\t\t\twindow.pyloadCaptchaOnLoadCallback = function() {\n\t\t\t\tvar widgetID = hcaptcha.render (\n\t\t\t\t\t"captchadiv",\n\t\t\t\t\t{size: "compact",\n\t\t\t\t\t \'sitekey\': gpyload.data.sitekey,\n\t\t\t\t\t \'callback\': function() {\n\t\t\t\t\t\tvar hcaptchaResponse = hcaptcha.getResponse(widgetID); // get captcha response\n\t\t\t\t\t\tgpyload.submitResponse(hcaptchaResponse);\n\t\t\t\t\t }}\n\t\t\t\t);\n\t\t\t\tgpyload.activated();\n\t\t\t};\n\n\t\t\tif(typeof hcaptcha !== \'undefined\' && hcaptcha) {\n\t\t\t\twindow.pyloadCaptchaOnLoadCallback();\n\t\t\t} else {\n\t\t\t\tvar js_script = document.createElement(\'script\');\n\t\t\t\tjs_script.type = "text/javascript";\n\t\t\t\tjs_script.src = "//hcaptcha.com/1/api.js?onload=pyloadCaptchaOnLoadCallback&render=explicit";\n\t\t\t\tjs_script.async = true;\n\t\t\t\tdocument.getElementsByTagName(\'head\')[0].appendChild(js_script);\n\t\t\t}'

    def detect_key(self, data=None):
        if False:
            while True:
                i = 10
        html = data or self.retrieve_data()
        m = re.search(self.KEY_PATTERN, html)
        if m is not None:
            key = urllib.parse.unquote(m.group(1).strip())
            m = re.search(self.KEY_FORMAT_PATTERN, key)
            if m is not None:
                self.key = key
                self.log_debug('Key: {}'.format(self.key))
                return self.key
            else:
                self.log_debug(key, 'Wrong key format, this probably because it is not a hCaptcha key')
        self.log_warning(self._('Key pattern not found'))
        return None

    def challenge(self, key=None, data=None):
        if False:
            print('Hello World!')
        key = key or self.retrieve_key(data)
        return self._challenge_js(key)

    def _challenge_js(self, key):
        if False:
            for i in range(10):
                print('nop')
        self.log_debug('Challenge hCaptcha interactive')
        params = {'url': self.pyfile.url, 'sitekey': key, 'script': {'signature': self.HCAPTCHA_INTERACTIVE_SIG, 'code': self.HCAPTCHA_INTERACTIVE_JS}}
        result = self.decrypt_interactive(params, timeout=300)
        return result
if __name__ == '__main__':
    import sys
    from ..helpers import sign_string
    if len(sys.argv) > 2:
        with open(sys.argv[1]) as fp:
            pem_private = fp.read()
        print(sign_string(HCaptcha.HCAPTCHA_INTERACTIVE_JS, pem_private, pem_passphrase=sys.argv[2], sign_algo='SHA384'))