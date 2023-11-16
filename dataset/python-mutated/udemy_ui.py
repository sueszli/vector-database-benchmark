"""Udemy UI."""
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List
from price_parser import Price
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver, WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from udemy_enroller.exceptions import LoginException, RobotException
from udemy_enroller.logger import get_logger
from udemy_enroller.settings import Settings
logger = get_logger()

@dataclass(unsafe_hash=True)
class RunStatistics:
    """Gather statistics on courses enrolled in."""
    prices: List[Decimal] = field(default_factory=list)
    expired: int = 0
    enrolled: int = 0
    already_enrolled: int = 0
    unwanted_language: int = 0
    unwanted_category: int = 0
    start_time = None
    currency_symbol = None

    def savings(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Calculate the savings made from enrolling to these courses.'
        return sum(self.prices) or 0

    def table(self):
        if False:
            print('Hello World!')
        'Log table of statistics to output.'
        if self.prices:
            if self.currency_symbol is None:
                self.currency_symbol = '¤'
            run_time_seconds = int((datetime.utcnow() - self.start_time).total_seconds())
            logger.info('==================Run Statistics==================')
            logger.info(f'Enrolled:                   {self.enrolled}')
            logger.info(f'Unwanted Category:          {self.unwanted_category}')
            logger.info(f'Unwanted Language:          {self.unwanted_language}')
            logger.info(f'Already Claimed:            {self.already_enrolled}')
            logger.info(f'Expired:                    {self.expired}')
            logger.info(f'Savings:                    {self.currency_symbol}{self.savings():.2f}')
            logger.info(f'Total run time (seconds):   {run_time_seconds}s')
            logger.info('==================Run Statistics==================')

class UdemyStatus(Enum):
    """Possible statuses of udemy course."""
    ALREADY_ENROLLED = 'ALREADY_ENROLLED'
    ENROLLED = 'ENROLLED'
    EXPIRED = 'EXPIRED'
    UNWANTED_LANGUAGE = 'UNWANTED_LANGUAGE'
    UNWANTED_CATEGORY = 'UNWANTED_CATEGORY'

class UdemyActionsUI:
    """Contains any logic related to interacting with udemy website."""
    DOMAIN = 'https://www.udemy.com'

    def __init__(self, driver: WebDriver, settings: Settings):
        if False:
            return 10
        'Initialize.'
        self.driver = driver
        self.settings = settings
        self.logged_in = False
        self.stats = RunStatistics()
        self.stats.start_time = datetime.utcnow()

    def login(self, is_retry=False) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Login to your udemy account.\n\n        :param bool is_retry: Is this is a login retry and we still have captcha raise RobotException\n\n        :return: None\n        '
        if not self.logged_in:
            self.driver.get(f'{self.DOMAIN}/join/login-popup/')
            if self.settings.email is None:
                self.settings.prompt_email()
            if self.settings.password is None:
                self.settings.prompt_password()
            try:
                email_element = self.driver.find_element_by_name('email')
                email_element.send_keys(self.settings.email)
                password_element = self.driver.find_element_by_name('password')
                password_element.send_keys(self.settings.password)
                self.driver.find_element_by_name('submit').click()
            except NoSuchElementException as e:
                is_robot = self._check_if_robot()
                if is_robot and (not is_retry):
                    input('Before login. Please solve the captcha before proceeding. Hit enter once solved ')
                    self.login(is_retry=True)
                    return
                if is_robot and is_retry:
                    raise RobotException('I am a bot!')
                raise e
            else:
                user_dropdown_xpath = "//a[@data-purpose='user-dropdown']"
                try:
                    WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, user_dropdown_xpath)))
                except TimeoutException:
                    is_robot = self._check_if_robot()
                    if is_robot and (not is_retry):
                        input('After login. Please solve the captcha before proceeding. Hit enter once solved ')
                        if self._check_if_robot():
                            raise RobotException('I am a bot!')
                        self.logged_in = True
                        return
                    raise LoginException('Udemy user failed to login')
                self.logged_in = True

    def enroll(self, url: str) -> str:
        if False:
            while True:
                i = 10
        '\n        Redeems the course url passed in.\n\n        :param str url: URL of the course to redeem\n        :return: A string detailing course status\n        '
        self.driver.get(url)
        course_name = self.driver.title
        if not self._check_languages(course_name):
            return UdemyStatus.UNWANTED_LANGUAGE.value
        if not self._check_categories(course_name):
            return UdemyStatus.UNWANTED_CATEGORY.value
        time.sleep(2)
        buy_course_button_xpath = "//button[@data-purpose='buy-this-course-button']"
        WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH, buy_course_button_xpath)))
        if not self._check_enrolled(course_name):
            element_present = EC.presence_of_element_located((By.XPATH, buy_course_button_xpath))
            WebDriverWait(self.driver, 10).until(element_present).click()
            enroll_button_xpath = "//div[starts-with(@class, 'checkout-button--checkout-button--container')]//button"
            element_present = EC.presence_of_element_located((By.XPATH, enroll_button_xpath))
            WebDriverWait(self.driver, 10).until(element_present)
            if self.settings.zip_code:
                try:
                    element_present = EC.presence_of_element_located((By.ID, 'billingAddressSecondaryInput'))
                    WebDriverWait(self.driver, 5).until(element_present).send_keys(self.settings.zip_code)
                    enroll_button_is_clickable = EC.element_to_be_clickable((By.XPATH, enroll_button_xpath))
                    WebDriverWait(self.driver, 5).until(enroll_button_is_clickable)
                except (TimeoutException, NoSuchElementException):
                    pass
            price_class_loading = 'udi-circle-loader'
            WebDriverWait(self.driver, 10).until_not(EC.presence_of_element_located((By.CLASS_NAME, price_class_loading)))
            if not self._check_price(course_name):
                return UdemyStatus.EXPIRED.value
            billing_state_element_id = 'billingAddressSecondarySelect'
            billing_state_elements = self.driver.find_elements_by_id(billing_state_element_id)
            if billing_state_elements:
                billing_state_elements[0].click()
                first_state_xpath = "//select[@id='billingAddressSecondarySelect']//option[2]"
                element_present = EC.presence_of_element_located((By.XPATH, first_state_xpath))
                WebDriverWait(self.driver, 10).until(element_present).click()
            enroll_button_is_clickable = EC.element_to_be_clickable((By.XPATH, enroll_button_xpath))
            WebDriverWait(self.driver, 10).until(enroll_button_is_clickable).click()
            success_element_class = "//div[contains(@class, 'success-alert-banner-container')]"
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, success_element_class)))
            logger.info(f"Successfully enrolled in: '{course_name}'")
            self.stats.enrolled += 1
            return UdemyStatus.ENROLLED.value
        else:
            return UdemyStatus.ALREADY_ENROLLED.value

    def _check_enrolled(self, course_name):
        if False:
            for i in range(10):
                print('nop')
        add_to_cart_xpath = "//div[starts-with(@class, 'buy-box')]//div[@data-purpose='add-to-cart']"
        add_to_cart_elements = self.driver.find_elements_by_xpath(add_to_cart_xpath)
        if not add_to_cart_elements or (add_to_cart_elements and (not add_to_cart_elements[0].is_displayed())):
            logger.debug(f"Already enrolled in '{course_name}'")
            self.stats.already_enrolled += 1
            return True
        return False

    def _check_languages(self, course_identifier):
        if False:
            return 10
        is_valid_language = True
        if self.settings.languages:
            locale_xpath = "//div[@data-purpose='lead-course-locale']"
            element_text = WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, locale_xpath))).text
            if element_text not in self.settings.languages:
                logger.debug(f'Course language not wanted: {element_text}')
                logger.debug(f"Course '{course_identifier}' language not wanted: {element_text}")
                self.stats.unwanted_language += 1
                is_valid_language = False
        return is_valid_language

    def _check_categories(self, course_identifier):
        if False:
            while True:
                i = 10
        is_valid_category = True
        if self.settings.categories:
            breadcrumbs_path = 'udlite-breadcrumb'
            breadcrumbs_text_path = 'udlite-heading-sm'
            breadcrumbs: WebElement = self.driver.find_element_by_class_name(breadcrumbs_path)
            breadcrumb_elements = breadcrumbs.find_elements_by_class_name(breadcrumbs_text_path)
            breadcrumb_text = [bc.text for bc in breadcrumb_elements]
            for category in self.settings.categories:
                if category in breadcrumb_text:
                    is_valid_category = True
                    break
            else:
                logger.debug(f"Skipping course '{course_identifier}' as it does not have a wanted category")
                self.stats.unwanted_category += 1
                is_valid_category = False
        return is_valid_category

    def _check_price(self, course_name):
        if False:
            i = 10
            return i + 15
        course_is_free = True
        price_xpath = "//div[contains(@data-purpose, 'total-amount-summary')]//span[2]"
        price_element = self.driver.find_element_by_xpath(price_xpath)
        if price_element.is_displayed():
            _price = price_element.text
            checkout_price = Price.fromstring(_price)
            if self.stats.currency_symbol is None and checkout_price.currency is not None:
                self.stats.currency_symbol = checkout_price.currency
            if checkout_price.amount is None or checkout_price.amount > 0:
                logger.debug(f"Skipping course '{course_name}' as it now costs {_price}")
                self.stats.expired += 1
                course_is_free = False
        if course_is_free:
            list_price_xpath = "//div[starts-with(@class, 'order-summary--original-price-text')]//span"
            list_price_element = self.driver.find_element_by_xpath(list_price_xpath)
            list_price = Price.fromstring(list_price_element.text)
            if list_price.amount is not None:
                self.stats.prices.append(list_price.amount)
        return course_is_free

    def _check_if_robot(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Simply checks if the captcha element is present on login if email/password elements are not.\n\n        :return: Bool\n        '
        is_robot = True
        try:
            self.driver.find_element_by_id('px-captcha')
        except NoSuchElementException:
            is_robot = False
        return is_robot