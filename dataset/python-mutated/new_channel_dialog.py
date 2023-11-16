from tribler.gui.defs import BUTTON_TYPE_CONFIRM, BUTTON_TYPE_NORMAL
from tribler.gui.dialogs.confirmationdialog import ConfirmationDialog
from tribler.gui.utilities import connect, tr

class NewChannelDialog(ConfirmationDialog):

    def __init__(self, parent, create_channel_callback):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent, tr('Create new channel'), tr('Enter the name of the channel/folder to create:'), [(tr('NEW'), BUTTON_TYPE_NORMAL), (tr('CANCEL'), BUTTON_TYPE_CONFIRM)], show_input=True)
        self.create_channel_callback = create_channel_callback
        self.dialog_widget.dialog_input.setPlaceholderText(tr('Channel name'))
        self.dialog_widget.dialog_input.setFocus()
        connect(self.button_clicked, self.on_channel_name_dialog_done)
        self.show()

    def on_channel_name_dialog_done(self, action):
        if False:
            i = 10
            return i + 15
        if action == 0:
            text = self.dialog_widget.dialog_input.text()
            if text:
                self.create_channel_callback(channel_name=text)
        self.close_dialog()