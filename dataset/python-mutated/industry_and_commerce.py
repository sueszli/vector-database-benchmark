import time
from geetest import BaseGeetestCrack
from selenium import webdriver

class IndustryAndCommerceGeetestCrack(BaseGeetestCrack):
    """工商滑动验证码破解类"""

    def __init__(self, driver):
        if False:
            i = 10
            return i + 15
        super(IndustryAndCommerceGeetestCrack, self).__init__(driver)

    def crack(self):
        if False:
            for i in range(10):
                print('nop')
        '执行破解程序\n\n        '
        self.input_by_id()
        self.click_by_id()
        x_offset = self.calculate_slider_offset()
        self.drag_and_drop(x_offset=x_offset)

def main():
    if False:
        while True:
            i = 10
    driver = webdriver.PhantomJS()
    driver.get('http://gsxt.hljaic.gov.cn/index.jspx')
    cracker = IndustryAndCommerceGeetestCrack(driver)
    cracker.crack()
    print(driver.get_window_size())
    time.sleep(10)
    driver.save_screenshot('screen.png')
    driver.close()
if __name__ == '__main__':
    main()