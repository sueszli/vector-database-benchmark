from picard.config import get_config
from picard.ui import PicardDialog
from picard.ui.ui_passworddialog import Ui_PasswordDialog

class PasswordDialog(PicardDialog):

    def __init__(self, authenticator, reply, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self._authenticator = authenticator
        self.ui = Ui_PasswordDialog()
        self.ui.setupUi(self)
        self.ui.info_text.setText(_('The server %s requires you to login. Please enter your username and password.') % reply.url().host())
        self.ui.username.setText(reply.url().userName())
        self.ui.password.setText(reply.url().password())
        self.ui.buttonbox.accepted.connect(self.set_new_password)

    def set_new_password(self):
        if False:
            while True:
                i = 10
        self._authenticator.setUser(self.ui.username.text())
        self._authenticator.setPassword(self.ui.password.text())
        self.accept()

class ProxyDialog(PicardDialog):

    def __init__(self, authenticator, proxy, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self._authenticator = authenticator
        self._proxy = proxy
        self.ui = Ui_PasswordDialog()
        self.ui.setupUi(self)
        config = get_config()
        self.ui.info_text.setText(_('The proxy %s requires you to login. Please enter your username and password.') % config.setting['proxy_server_host'])
        self.ui.username.setText(config.setting['proxy_username'])
        self.ui.password.setText(config.setting['proxy_password'])
        self.ui.buttonbox.accepted.connect(self.set_proxy_password)

    def set_proxy_password(self):
        if False:
            for i in range(10):
                print('nop')
        config = get_config()
        config.setting['proxy_username'] = self.ui.username.text()
        config.setting['proxy_password'] = self.ui.password.text()
        self._authenticator.setUser(self.ui.username.text())
        self._authenticator.setPassword(self.ui.password.text())
        self.accept()