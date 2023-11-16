from selenium.webdriver.common.by import By
from .base import BasePage, ButtonElement, ModalElement

class ExampleIntegrationSetupWindowElement(ModalElement):
    name_field_selector = 'name'
    submit_button_selector = '[type="submit"]'

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self.name = self.element.find_element(by=By.NAME, value='name')
        continue_button_element = self.element.find_element(by=By.CSS_SELECTOR, value=self.submit_button_selector)
        self.continue_button = ButtonElement(continue_button_element)

    def fill_in_setup_form(self, installation_data):
        if False:
            i = 10
            return i + 15
        self.name.send_keys(installation_data[self.name_field_selector])

class OrganizationAbstractDetailViewPage(BasePage):
    configurations_text = 'Configurations'

    def click_install_button(self):
        if False:
            return 10
        self.browser.click('[data-test-id="install-button"]')

    def uninstall(self):
        if False:
            for i in range(10):
                print('nop')
        self.browser.click('[aria-label="Uninstall"]')
        self.browser.click('[data-test-id="confirm-button"]')

    def switch_to_configuration_view(self):
        if False:
            print('Hello World!')
        self.browser.find_element(by=By.LINK_TEXT, value=self.configurations_text).click()

class OrganizationIntegrationDetailViewPage(OrganizationAbstractDetailViewPage):

    def click_through_integration_setup(self, setup_window_cls, installation_data):
        if False:
            return 10
        self.driver.switch_to.window(self.driver.window_handles[1])
        integration_setup_window = setup_window_cls(element=self.browser)
        integration_setup_window.fill_in_setup_form(installation_data)
        integration_setup_window.continue_button.click()
        self.driver.switch_to.window(self.driver.window_handles[0])

class OrganizationSentryAppDetailViewPage(OrganizationAbstractDetailViewPage):

    def uninstall(self):
        if False:
            while True:
                i = 10
        self.browser.click('[data-test-id="sentry-app-uninstall"]')
        self.browser.click('[data-test-id="confirm-button"]')