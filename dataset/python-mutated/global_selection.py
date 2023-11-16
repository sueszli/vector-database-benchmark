from .base import BasePage

class GlobalSelectionPage(BasePage):

    def get_selected_project_slug(self):
        if False:
            i = 10
            return i + 15
        return self.browser.element('[data-test-id="page-filter-project-selector"]').text

    def get_selected_environment(self):
        if False:
            i = 10
            return i + 15
        return self.browser.element('[data-test-id="page-filter-environment-selector"]').text

    def get_selected_date(self):
        if False:
            print('Hello World!')
        return self.browser.element('[data-test-id="page-filter-timerange-selector"]').text

    def go_back_to_issues(self):
        if False:
            i = 10
            return i + 15
        self.browser.click('[data-test-id="breadcrumb-link"]')

    def open_project_selector(self):
        if False:
            i = 10
            return i + 15
        self.browser.click('[data-test-id="page-filter-project-selector"]')

    def select_project_by_slug(self, slug):
        if False:
            print('Hello World!')
        project_item_selector = f'//*[@data-test-id="menu-list-item-label" and text()="{slug}"]'
        self.open_project_selector()
        self.browser.wait_until(xpath=project_item_selector)
        self.browser.click(xpath=project_item_selector)

    def open_environment_selector(self):
        if False:
            return 10
        self.browser.click('[data-test-id="page-filter-environment-selector"]')

    def select_environment(self, environment):
        if False:
            i = 10
            return i + 15
        environment_path = f'//*[@data-test-id="menu-list-item-label" and text()="{environment}"]'
        self.open_environment_selector()
        self.browser.wait_until(xpath=environment_path)
        self.browser.click(xpath=environment_path)

    def open_date_selector(self):
        if False:
            while True:
                i = 10
        self.browser.click('[data-test-id="page-filter-timerange-selector"]')

    def select_date(self, date):
        if False:
            print('Hello World!')
        date_path = f'//*[text()="{date}"]'
        self.open_date_selector()
        self.browser.wait_until(xpath=date_path)
        self.browser.click(xpath=date_path)