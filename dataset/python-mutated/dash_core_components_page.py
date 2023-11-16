import logging
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
logger = logging.getLogger(__name__)

class DashCoreComponentsMixin(object):

    def select_date_single(self, compid, index=0, day='', outside_month=False):
        if False:
            i = 10
            return i + 15
        'Select Date in DPS component with either index or day\n        compid: the id defined for component\n        index: the index for all visibles in the popup calendar\n        day: a number or string; if set, use this to select instead of index\n        outside_month: used in conjunction with day. indicates if the day out\n            the scope of current month. default False.\n        '
        date = self.find_element(f'#{compid} input')
        date.click()

        def is_month_valid(elem):
            if False:
                for i in range(10):
                    print('nop')
            return '__outside' in elem.get_attribute('class') if outside_month else '__outside' not in elem.get_attribute('class')
        self._wait_until_day_is_clickable()
        days = self.find_elements(self.date_picker_day_locator)
        if day:
            filtered = [_ for _ in days if _.text == str(day) and is_month_valid(_)]
            if not filtered or len(filtered) > 1:
                logger.error('cannot find the matched day with index=%s, day=%s', index, day)
            matched = filtered[0]
        else:
            matched = days[index]
        matched.click()
        return date.get_attribute('value')

    def select_date_range(self, compid, day_range, start_first=True):
        if False:
            return 10
        'Select Date in DPR component with a day_range tuple\n        compid: the id defined for component\n        day_range: a tuple or list, defines the start and end date you want to\n            select, the tuple must be length of 1 or 2, i.e.\n            (start, ) or (start, end)\n        start_first: boolean value decides clicking start or end date.\n            default True\n        '
        if not day_range or not isinstance(day_range, (tuple, list)) or (not set(range(1, 32)).issuperset(day_range)) or (len(day_range) > 2):
            logger.error('data_range is provided with an invalid value %s\nthe accepted range is range(1, 32)', day_range)
            return
        prefix = 'Start' if start_first else 'End'
        date = self.find_element(f'#{compid} input[aria-label="{prefix} Date"]')
        date.click()
        for day in day_range:
            self._wait_until_day_is_clickable()
            matched = [_ for _ in self.find_elements(self.date_picker_day_locator) if _.text == str(day)]
            matched[0].click()
        return self.get_date_range(compid)

    def get_date_range(self, compid):
        if False:
            while True:
                i = 10
        return tuple((_.get_attribute('value') for _ in self.find_elements(f'#{compid} input')))

    def _wait_until_day_is_clickable(self, timeout=1):
        if False:
            while True:
                i = 10
        WebDriverWait(self.driver, timeout).until(EC.element_to_be_clickable((By.CSS_SELECTOR, self.date_picker_day_locator)))

    @property
    def date_picker_day_locator(self):
        if False:
            print('Hello World!')
        return 'div[data-visible="true"] td.CalendarDay'

    def click_and_hold_at_coord_fractions(self, elem_or_selector, fx, fy):
        if False:
            for i in range(10):
                print('nop')
        elem = self._get_element(elem_or_selector)
        ActionChains(self.driver).move_to_element_with_offset(elem, elem.size['width'] * fx, elem.size['height'] * fy).click_and_hold().perform()

    def move_to_coord_fractions(self, elem_or_selector, fx, fy):
        if False:
            print('Hello World!')
        elem = self._get_element(elem_or_selector)
        ActionChains(self.driver).move_to_element_with_offset(elem, elem.size['width'] * fx, elem.size['height'] * fy).perform()

    def release(self):
        if False:
            while True:
                i = 10
        ActionChains(self.driver).release().perform()

    def click_and_drag_at_coord_fractions(self, elem_or_selector, fx1, fy1, fx2, fy2):
        if False:
            while True:
                i = 10
        elem = self._get_element(elem_or_selector)
        ActionChains(self.driver).move_to_element_with_offset(elem, elem.size['width'] * fx1, elem.size['height'] * fy1).click_and_hold().move_to_element_with_offset(elem, elem.size['width'] * fx2, elem.size['height'] * fy2).release().perform()