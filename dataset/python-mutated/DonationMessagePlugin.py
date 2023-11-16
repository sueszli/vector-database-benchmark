import re
from Plugin import PluginManager

@PluginManager.registerTo('UiRequest')
class UiRequestPlugin(object):

    def renderWrapper(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        body = super(UiRequestPlugin, self).renderWrapper(*args, **kwargs)
        inject_html = "\n            <style>\n             #donation_message { position: absolute; bottom: 0px; right: 20px; padding: 7px; font-family: Arial; font-size: 11px }\n            </style>\n            <a id='donation_message' href='https://blockchain.info/address/1QDhxQ6PraUZa21ET5fYUCPgdrwBomnFgX' target='_blank'>Please donate to help to keep this ZeroProxy alive</a>\n            </body>\n            </html>\n        "
        return re.sub('</body>\\s*</html>\\s*$', inject_html, body)