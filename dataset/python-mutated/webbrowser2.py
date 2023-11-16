"""A webbrowser extension for Picard.

It handles and displays errors in PyQt and also adds a utility function for opening Picard URLS.
"""
import webbrowser
from PyQt6 import QtWidgets
from picard.const import PICARD_URLS

def open(url):
    if False:
        return 10
    if url in PICARD_URLS:
        url = PICARD_URLS[url]
    try:
        webbrowser.open(url)
    except webbrowser.Error as e:
        QtWidgets.QMessageBox.critical(None, _('Web Browser Error'), _('Error while launching a web browser:\n\n%s') % (e,))