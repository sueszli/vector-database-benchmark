"""Convenience functions to show message boxes."""
from qutebrowser.qt.core import Qt
from qutebrowser.qt.widgets import QMessageBox
from qutebrowser.misc import objects
from qutebrowser.utils import log

class DummyBox:
    """A dummy QMessageBox returned when --no-err-windows is used."""

    def exec(self):
        if False:
            print('Hello World!')
        pass

def msgbox(parent, title, text, *, icon, buttons=QMessageBox.StandardButton.Ok, on_finished=None, plain_text=None):
    if False:
        return 10
    "Display a QMessageBox with the given icon.\n\n    Args:\n        parent: The parent to set for the message box.\n        title: The title to set.\n        text: The text to set.\n        icon: The QIcon to show in the box.\n        buttons: The buttons to set (QMessageBox::StandardButtons)\n        on_finished: A slot to connect to the 'finished' signal.\n        plain_text: Whether to force plain text (True) or rich text (False).\n                    None (the default) uses Qt's auto detection.\n\n    Return:\n        A new QMessageBox.\n    "
    if objects.args.no_err_windows:
        log.misc.info(f'{title}\n\n{text}')
        return DummyBox()
    box = QMessageBox(parent)
    box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    box.setIcon(icon)
    box.setStandardButtons(buttons)
    if on_finished is not None:
        box.finished.connect(on_finished)
    if plain_text:
        box.setTextFormat(Qt.TextFormat.PlainText)
    elif plain_text is not None:
        box.setTextFormat(Qt.TextFormat.RichText)
    box.setWindowTitle(title)
    box.setText(text)
    box.show()
    return box

def information(*args, **kwargs):
    if False:
        return 10
    'Display an information box.\n\n    Args:\n        *args: Passed to msgbox.\n        **kwargs: Passed to msgbox.\n\n    Return:\n        A new QMessageBox.\n    '
    return msgbox(*args, icon=QMessageBox.Icon.Information, **kwargs)