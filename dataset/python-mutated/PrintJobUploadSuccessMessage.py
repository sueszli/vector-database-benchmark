from UM import i18nCatalog
from UM.Message import Message
I18N_CATALOG = i18nCatalog('cura')

class PrintJobUploadSuccessMessage(Message):
    """Message shown when uploading a print job to a cluster succeeded."""

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        super().__init__(text=I18N_CATALOG.i18nc('@info:status', 'Print job was successfully sent to the printer.'), title=I18N_CATALOG.i18nc('@info:title', 'Data Sent'), message_type=Message.MessageType.POSITIVE)