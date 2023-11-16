from PyQt5.QtCore import Q_CLASSINFO, pyqtSlot
from PyQt5.QtDBus import QDBusAbstractAdaptor, QDBusConnection

class AppService(QDBusAbstractAdaptor):
    Q_CLASSINFO('D-Bus Interface', 'org.autokey.Service')
    Q_CLASSINFO('D-Bus Introspection', '  <interface name="org.autokey.Service">\n    <method name="show_configure"/>\n    <method name="run_script">\n      <arg type="s" name="name" direction="in"/>\n    </method>\n    <method name="run_phrase">\n      <arg type="s" name="name" direction="in"/>\n    </method>\n    <method name="run_folder">\n      <arg type="s" name="name" direction="in"/>\n    </method>\n  </interface>\n')

    def __init__(self, parent):
        if False:
            while True:
                i = 10
        super(AppService, self).__init__(parent)
        self.connection = QDBusConnection.sessionBus()
        path = '/AppService'
        service = 'org.autokey.Service'
        self.connection.registerObject(path, parent)
        self.connection.registerService(service)
        self.setAutoRelaySignals(True)

    @pyqtSlot()
    def show_configure(self):
        if False:
            print('Hello World!')
        self.parent().show_configure()

    @pyqtSlot(str)
    def run_script(self, name):
        if False:
            while True:
                i = 10
        self.parent().service.run_script(name)

    @pyqtSlot(str)
    def run_phrase(self, name):
        if False:
            for i in range(10):
                print('nop')
        self.parent().service.run_phrase(name)

    @pyqtSlot(str)
    def run_folder(self, name):
        if False:
            return 10
        self.parent().service.run_folder(name)