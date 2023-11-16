from .WelcomePagesModel import WelcomePagesModel

class AddPrinterPagesModel(WelcomePagesModel):

    def initialize(self, cancellable: bool=True) -> None:
        if False:
            while True:
                i = 10
        self._pages.append({'id': 'add_network_or_local_printer', 'page_url': self._getBuiltinWelcomePagePath('AddUltimakerOrThirdPartyPrinterStack.qml'), 'next_page_id': 'machine_actions', 'next_page_button_text': self._catalog.i18nc('@action:button', 'Add')})
        self._pages.append({'id': 'add_printer_by_ip', 'page_url': self._getBuiltinWelcomePagePath('AddPrinterByIpContent.qml'), 'next_page_id': 'machine_actions'})
        self._pages.append({'id': 'add_cloud_printers', 'page_url': self._getBuiltinWelcomePagePath('AddCloudPrintersView.qml'), 'is_final_page': True, 'next_page_button_text': self._catalog.i18nc('@action:button', 'Finish')})
        self._pages.append({'id': 'machine_actions', 'page_url': self._getBuiltinWelcomePagePath('FirstStartMachineActionsContent.qml'), 'should_show_function': self.shouldShowMachineActions})
        if cancellable:
            self._pages[0]['previous_page_button_text'] = self._catalog.i18nc('@action:button', 'Cancel')
        self.setItems(self._pages)
__all__ = ['AddPrinterPagesModel']