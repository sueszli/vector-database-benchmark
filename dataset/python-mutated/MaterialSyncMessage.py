from typing import TYPE_CHECKING
from UM import i18nCatalog
from UM.Message import Message
if TYPE_CHECKING:
    from ..UltimakerNetworkedPrinterOutputDevice import UltimakerNetworkedPrinterOutputDevice
I18N_CATALOG = i18nCatalog('cura')

class MaterialSyncMessage(Message):
    """Message shown when sending material files to cluster host."""
    __is_visible = False
    'Singleton used to prevent duplicate messages of this type at the same time.'

    def __init__(self, device: 'UltimakerNetworkedPrinterOutputDevice') -> None:
        if False:
            return 10
        super().__init__(text=I18N_CATALOG.i18nc('@info:status', 'Cura has detected material profiles that were not yet installed on the host printer of group {0}.', device.name), title=I18N_CATALOG.i18nc('@info:title', 'Sending materials to printer'), lifetime=10, dismissable=True)

    def show(self) -> None:
        if False:
            while True:
                i = 10
        if MaterialSyncMessage.__is_visible:
            return
        super().show()
        MaterialSyncMessage.__is_visible = True

    def hide(self, send_signal=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().hide(send_signal)
        MaterialSyncMessage.__is_visible = False