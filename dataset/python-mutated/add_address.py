import sys
from PyQt4.QtGui import QApplication
from PyQt4.QtCore import QUrl
from PyQt4.QtWebKit import QWebView
from PyQt4.QtGui import QGridLayout, QLineEdit, QWidget

class UrlInput(QLineEdit):

    def __init__(self, browser):
        if False:
            i = 10
            return i + 15
        super(UrlInput, self).__init__()
        self.browser = browser
        self.returnPressed.connect(self._return_pressed)

    def _return_pressed(self):
        if False:
            print('Hello World!')
        url = QUrl(self.text())
        browser.load(url)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    grid = QGridLayout()
    browser = QWebView()
    url_input = UrlInput(browser)
    grid.addWidget(url_input, 1, 0)
    grid.addWidget(browser, 2, 0)
    main_frame = QWidget()
    main_frame.setLayout(grid)
    main_frame.show()
    sys.exit(app.exec_())