""" Module that handles the like features """
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from .util import update_activity
LIKE_TAG_CLASS = 'coreSpriteHeartOpen'

def get_like_on_feed(browser, amount):
    if False:
        for i in range(10):
            print('nop')
    '\n    browser - the selenium browser element\n    amount - total amount of likes to perform\n\n    --------------------------------------\n    The function takes in the total amount of likes to perform\n    and then sends buttons to be liked, if it has run out of like\n    buttons it will perform a scroll\n    '
    assert 1 <= amount
    likes_performed = 0
    while likes_performed != amount:
        try:
            like_buttons = browser.find_elements(By.CLASS_NAME, LIKE_TAG_CLASS)
        except NoSuchElementException:
            print('Unable to find the like buttons, aborting')
            break
        else:
            for button in like_buttons:
                likes_performed += 1
                if amount < likes_performed:
                    print('Performed the required number of likes')
                    break
                yield button
            print('--> Total Likes uptil now ->', likes_performed)
            browser.execute_script('window.scrollTo(0, document.body.scrollHeight);')
            update_activity(browser, state=None)