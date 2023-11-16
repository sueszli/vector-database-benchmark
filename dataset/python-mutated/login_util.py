"""Module only used for the login part of the script"""
import json
import os
import pickle
import random
import socket
from selenium.common.exceptions import MoveTargetOutOfBoundsException, NoSuchElementException, WebDriverException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from .time_util import sleep
from .util import check_authorization, click_element, explicit_wait, reload_webpage, update_activity, web_address_navigator
from .xpath import read_xpath

def bypass_suspicious_login(browser, logger, logfolder, bypass_security_challenge_using):
    if False:
        return 10
    'Bypass suspicious loggin attempt verification.'
    dismiss_get_app_offer(browser, logger)
    dismiss_notification_offer(browser, logger)
    dismiss_this_was_me(browser)
    option = None
    if bypass_security_challenge_using == 'sms':
        try:
            option = browser.find_element(By.XPATH, read_xpath(bypass_suspicious_login.__name__, 'bypass_with_sms_option'))
        except NoSuchElementException:
            logger.warning('Unable to choose ({}) option to bypass the challenge'.format(bypass_security_challenge_using.upper()))
    if bypass_security_challenge_using == 'email':
        try:
            option = browser.find_element(By.XPATH, read_xpath(bypass_suspicious_login.__name__, 'bypass_with_email_option'))
        except NoSuchElementException:
            logger.warning('Unable to choose ({}) option to bypass the challenge'.format(bypass_security_challenge_using.upper()))
    ActionChains(browser).move_to_element(option).click().perform()
    option_text = option.text
    send_security_code_button = browser.find_element(By.XPATH, read_xpath(bypass_suspicious_login.__name__, 'send_security_code_button'))
    ActionChains(browser).move_to_element(send_security_code_button).click().perform()
    update_activity(browser, state=None)
    logger.info('Instagram detected an unusual login attempt')
    logger.info('Check Instagram App for "Suspicious Login attempt" prompt')
    logger.info('A security code was sent to your {}'.format(option_text))
    security_code = None
    try:
        path = '{}state.json'.format(logfolder)
        data = {}
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            with open(path, 'r') as json_file:
                data = json.load(json_file)
        security_code = data['challenge']['security_code']
    except Exception:
        logger.info('Security Code not present in {}state.json file'.format(logfolder))
    if security_code is None:
        security_code = input('Type the security code here: ')
    security_code_field = browser.find_element(By.XPATH, read_xpath(bypass_suspicious_login.__name__, 'security_code_field'))
    ActionChains(browser).move_to_element(security_code_field).click().send_keys(security_code).perform()
    for _ in range(2):
        update_activity(browser, state=None)
    submit_security_code_button = browser.find_element(By.XPATH, read_xpath(bypass_suspicious_login.__name__, 'submit_security_code_button'))
    ActionChains(browser).move_to_element(submit_security_code_button).click().perform()
    update_activity(browser, state=None)
    try:
        sleep(3)
        wrong_login = browser.find_element(By.XPATH, read_xpath(bypass_suspicious_login.__name__, 'wrong_login'))
        if wrong_login is not None:
            wrong_login_msg = 'Wrong security code! Please check the code Instagram sent you and try again.'
            update_activity(browser, action=None, state=wrong_login_msg, logfolder=logfolder, logger=logger)
            logger.warning(wrong_login_msg)
    except NoSuchElementException:
        pass

def check_browser(browser, logfolder, logger, proxy_address):
    if False:
        return 10
    update_activity(browser, action=None, state='trying to connect', logfolder=logfolder, logger=logger)
    try:
        logger.info('-- Connection Checklist [1/2] (Internet Connection Status)')
        browser.get('view-source:https://freegeoip.app/json')
        pre = browser.find_element(By.TAG_NAME, 'pre').text
        current_ip_info = json.loads(pre)
        if proxy_address is not None and socket.gethostbyname(proxy_address) != current_ip_info['ip']:
            logger.warning("- Proxy is set, but it's not working properly")
            logger.warning('- Expected Proxy IP is "{}", and the current IP is "{}"'.format(proxy_address, current_ip_info['ip']))
            logger.warning('- Try again or disable the Proxy Address on your setup')
            logger.warning('- Aborting connection...')
            return False
        else:
            logger.info('- Internet Connection Status: ok')
            logger.info('- Current IP is "{}" and it\'s from "{}/{}"'.format(current_ip_info['ip'], current_ip_info['country_name'], current_ip_info['country_code']))
            update_activity(browser, action=None, state='Internet connection is ok', logfolder=logfolder, logger=logger)
    except Exception:
        logger.warning('- Internet Connection Status: error')
        update_activity(browser, action=None, state='There is an issue with the internet connection', logfolder=logfolder, logger=logger)
        return False
    logger.info('-- Connection Checklist [2/2] (Hide Selenium Extension)')
    webdriver = browser.execute_script('return window.navigator.webdriver')
    logger.info('- window.navigator.webdriver response: {}'.format(webdriver))
    if webdriver:
        logger.warning('- Hide Selenium Extension: error')
    else:
        logger.info('- Hide Selenium Extension: ok')
    return True

def login_user(browser, username, password, logger, logfolder, proxy_address, security_code_to_phone, security_codes, want_check_browser):
    if False:
        return 10
    'Logins the user with the given username and password'
    assert username, 'Username not provided'
    assert password, 'Password not provided'
    if want_check_browser:
        if not check_browser(browser, logfolder, logger, proxy_address):
            return False
    ig_homepage = 'https://www.instagram.com'
    web_address_navigator(browser, ig_homepage)
    cookie_file = '{0}{1}_cookie.pkl'.format(logfolder, username)
    cookie_loaded = None
    login_state = None
    try:
        for cookie in pickle.load(open(cookie_file, 'rb')):
            if 'sameSite' in cookie and cookie['sameSite'] == 'None':
                cookie['sameSite'] = 'Strict'
            browser.add_cookie(cookie)
        sleep(4)
        cookie_loaded = True
        logger.info("- Cookie file for user '{}' loaded...".format(username))
        reload_webpage(browser)
        sleep(4)
        login_state = check_authorization(browser, username, 'activity counts', logger, False)
        sleep(4)
    except (WebDriverException, OSError, IOError):
        logger.info('- Cookie file not found, creating cookie...')
    if login_state and cookie_loaded:
        dismiss_notification_offer(browser, logger)
        dismiss_save_information(browser, logger)
        accept_igcookie_dialogue(browser, logger)
        return True
    accept_igcookie_dialogue(browser, logger)
    if cookie_loaded:
        logger.warning("- Issue with cookie for user '{}'. Creating new cookie...".format(username))
        try:
            logger.info('- Deleting browser cookies...')
            browser.delete_all_cookies()
            browser.refresh()
            os.remove(cookie_file)
            sleep(random.randint(3, 5))
        except Exception as e:
            if isinstance(e, WebDriverException):
                logger.exception('Error occurred while deleting cookies from web browser!\n\t{}'.format(str(e).encode('utf-8')))
            return False
    web_address_navigator(browser, ig_homepage)
    try:
        login_elem = browser.find_element(By.XPATH, read_xpath(login_user.__name__, 'login_elem'))
    except NoSuchElementException:
        logger.warning('Login A/B test detected! Trying another string...')
        try:
            login_elem = browser.find_element(By.XPATH, read_xpath(login_user.__name__, 'login_elem_no_such_exception'))
        except NoSuchElementException:
            logger.warning('Could not pass the login A/B test. Trying last string...')
            try:
                login_elem = browser.find_element(By.XPATH, read_xpath(login_user.__name__, 'login_elem_no_such_exception_2'))
            except NoSuchElementException as e:
                logger.exception('Login A/B test failed!\n\t{}'.format(str(e).encode('utf-8')))
                return False
    if login_elem is not None:
        try:
            ActionChains(browser).move_to_element(login_elem).click().perform()
        except MoveTargetOutOfBoundsException:
            login_elem.click()
        update_activity(browser, state=None)
    login_page_title = 'Instagram'
    explicit_wait(browser, 'TC', login_page_title, logger)
    input_username_XP = read_xpath(login_user.__name__, 'input_username_XP')
    explicit_wait(browser, 'VOEL', [input_username_XP, 'XPath'], logger)
    input_username = browser.find_element(By.XPATH, input_username_XP)
    ActionChains(browser).move_to_element(input_username).click().send_keys(username).perform()
    for _ in range(2):
        update_activity(browser, state=None)
    sleep(1)
    input_password = browser.find_element(By.XPATH, read_xpath(login_user.__name__, 'input_password'))
    if not isinstance(password, str):
        password = str(password)
    ActionChains(browser).move_to_element(input_password).click().send_keys(password).perform()
    sleep(1)
    ActionChains(browser).move_to_element(input_password).click().send_keys(Keys.ENTER).perform()
    for _ in range(4):
        update_activity(browser, state=None)
    two_factor_authentication(browser, logger, security_codes)
    dismiss_get_app_offer(browser, logger)
    dismiss_notification_offer(browser, logger)
    dismiss_save_information(browser, logger)
    accept_igcookie_dialogue(browser, logger)
    if 'instagram.com/challenge' in browser.current_url:
        try:
            account_disabled = browser.find_element(By.XPATH, read_xpath(login_user.__name__, 'account_disabled'))
            logger.warning(account_disabled.text)
            update_activity(browser, action=None, state=account_disabled.text, logfolder=logfolder, logger=logger)
            return False
        except NoSuchElementException:
            pass
        try:
            browser.find_element(By.XPATH, read_xpath(login_user.__name__, 'add_phone_number'))
            challenge_warn_msg = "Instagram initiated a challenge before allow your account to login. At the moment there isn't a phone number linked to your Instagram account. Please, add a phone number to your account, and try again."
            logger.warning(challenge_warn_msg)
            update_activity(browser, action=None, state=challenge_warn_msg, logfolder=logfolder, logger=logger)
            return False
        except NoSuchElementException:
            pass
        try:
            browser.find_element(By.XPATH, read_xpath(login_user.__name__, 'suspicious_login_attempt'))
            update_activity(browser, action=None, state='Trying to solve suspicious attempt login', logfolder=logfolder, logger=logger)
            bypass_suspicious_login(browser, logger, logfolder, security_code_to_phone)
        except NoSuchElementException:
            pass
    try:
        error_alert = browser.find_element(By.XPATH, read_xpath(login_user.__name__, 'error_alert'))
        logger.warning(error_alert.text)
        update_activity(browser, action=None, state=error_alert.text, logfolder=logfolder, logger=logger)
        return False
    except NoSuchElementException:
        pass
    if 'instagram.com/accounts/onetap' in browser.current_url:
        browser.get(ig_homepage)
    explicit_wait(browser, 'PFL', [], logger, 5)
    nav = browser.find_element(By.XPATH, read_xpath(login_user.__name__, 'nav'))
    if nav is not None:
        cookies_list = browser.get_cookies()
        for cookie in cookies_list:
            if 'sameSite' in cookie and cookie['sameSite'] == 'None':
                cookie['sameSite'] = 'Strict'
        try:
            with open(cookie_file, 'wb') as cookie_f_handler:
                pickle.dump(cookies_list, cookie_f_handler)
        except pickle.PicklingError:
            logger.warning('- Browser cookie list could not be saved to your local...')
        finally:
            return True
    else:
        return False

def dismiss_get_app_offer(browser, logger):
    if False:
        i = 10
        return i + 15
    "Dismiss 'Get the Instagram App' page after a fresh login"
    offer_elem = read_xpath(dismiss_get_app_offer.__name__, 'offer_elem')
    dismiss_elem = read_xpath(dismiss_get_app_offer.__name__, 'dismiss_elem')
    offer_loaded = explicit_wait(browser, 'VOEL', [offer_elem, 'XPath'], logger, 5, False)
    if offer_loaded:
        dismiss_elem = browser.find_element(By.XPATH, dismiss_elem)
        click_element(browser, dismiss_elem)

def dismiss_notification_offer(browser, logger):
    if False:
        print('Hello World!')
    "Dismiss 'Turn on Notifications' offer on session start"
    offer_elem_loc = read_xpath(dismiss_notification_offer.__name__, 'offer_elem_loc')
    dismiss_elem_loc = read_xpath(dismiss_notification_offer.__name__, 'dismiss_elem_loc')
    offer_loaded = explicit_wait(browser, 'VOEL', [offer_elem_loc, 'XPath'], logger, 4, False)
    if offer_loaded:
        dismiss_elem = browser.find_element(By.XPATH, dismiss_elem_loc)
        click_element(browser, dismiss_elem)

def dismiss_save_information(browser, logger):
    if False:
        print('Hello World!')
    "Dismiss 'Save Your Login Info?' offer on session start"
    offer_elem_loc = read_xpath(dismiss_save_information.__name__, 'offer_elem_loc')
    dismiss_elem_loc = read_xpath(dismiss_save_information.__name__, 'dismiss_elem_loc')
    offer_loaded = explicit_wait(browser, 'VOEL', [offer_elem_loc, 'XPath'], logger, 4, False)
    if offer_loaded:
        logger.info('- Do not save Login Info by now...')
        dismiss_elem = browser.find_element(By.XPATH, dismiss_elem_loc)
        click_element(browser, dismiss_elem)

def dismiss_this_was_me(browser):
    if False:
        print('Hello World!')
    try:
        this_was_me_button = browser.find_element(By.XPATH, read_xpath(dismiss_this_was_me.__name__, 'this_was_me_button'))
        ActionChains(browser).move_to_element(this_was_me_button).click().perform()
        update_activity(browser, state=None)
    except NoSuchElementException:
        pass

def two_factor_authentication(browser, logger, security_codes):
    if False:
        while True:
            i = 10
    '\n    Check if account is protected with Two Factor Authentication codes\n\n    Args:\n        :browser: Web driver\n        :logger: Library to log actions\n        :security_codes: List of Two Factor Authentication codes, also named as Recovery Codes.\n\n    Returns: None\n    '
    sleep(random.randint(3, 5))
    if 'two_factor' in browser.current_url:
        logger.info('- Two Factor Authentication is enabled...')
        code = random.choice(security_codes)
        try:
            int(code)
            verification_code = read_xpath(login_user.__name__, 'verification_code')
            explicit_wait(browser, 'VOEL', [verification_code, 'XPath'], logger)
            security_code = browser.find_element(By.XPATH, verification_code)
            confirm = browser.find_element(By.XPATH, read_xpath(login_user.__name__, 'confirm'))
            ActionChains(browser).move_to_element(security_code).click().send_keys(code).perform()
            sleep(random.randint(1, 3))
            ActionChains(browser).move_to_element(confirm).click().send_keys(Keys.ENTER).perform()
            for _ in range(2):
                update_activity(browser, state=None)
            sleep(random.randint(1, 3))
        except NoSuchElementException as e:
            logger.warning('- Secuirty code could not be written!\n\t{}'.format(str(e).encode('utf-8')))
        except ValueError:
            logger.warning('- Secuirty code provided is not a number')
    else:
        return

def accept_igcookie_dialogue(browser, logger):
    if False:
        print('Hello World!')
    "Presses 'Accept' button on IG cookie dialogue"
    offer_elem_loc = read_xpath(accept_igcookie_dialogue.__name__, 'accept_button')
    offer_loaded = explicit_wait(browser, 'VOEL', [offer_elem_loc, 'XPath'], logger, 4, False)
    if offer_loaded:
        logger.info('- Accepted IG cookies by default...')
        accept_elem = browser.find_element(By.XPATH, offer_elem_loc)
        click_element(browser, accept_elem)