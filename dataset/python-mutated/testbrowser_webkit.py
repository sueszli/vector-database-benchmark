"""Very simple browser for testing purposes."""
import sys
import argparse
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWebKit import QWebSettings
from PyQt5.QtWebKitWidgets import QWebView

def parse_args():
    if False:
        for i in range(10):
            print('nop')
    'Parse commandline arguments.'
    parser = argparse.ArgumentParser()
    parser.add_argument('url', help='The URL to open')
    parser.add_argument('--plugins', '-p', help='Enable plugins', default=False, action='store_true')
    return parser.parse_known_args()[0]
if __name__ == '__main__':
    args = parse_args()
    app = QApplication(sys.argv)
    wv = QWebView()
    wv.loadStarted.connect(lambda : print('Loading started'))
    wv.loadProgress.connect(lambda p: print('Loading progress: {}%'.format(p)))
    wv.loadFinished.connect(lambda : print('Loading finished'))
    if args.plugins:
        wv.settings().setAttribute(QWebSettings.PluginsEnabled, True)
    wv.load(QUrl.fromUserInput(args.url))
    wv.show()
    app.exec()