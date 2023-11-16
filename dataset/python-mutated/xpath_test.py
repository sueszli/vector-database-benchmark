import unittest
from logzero import logger
import uiautomator2 as u2
d = u2.connect_usb()

class MusicTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.package_name = 'com.netease.cloudmusic'
        d.ext_xpath.global_set({'timeout': 10})
        logger.info('setUp unlock-screen')
        d.screen_on()
        d.shell('input keyevent HOME')
        d.swipe(0.1, 0.9, 0.9, 0.1)

    def runTest(self):
        if False:
            for i in range(10):
                print('nop')
        logger.info('runTest')
        d.app_clear(self.package_name)
        s = d.session(self.package_name)
        s.set_fastinput_ime(True)
        xp = d.ext_xpath
        xp._d = s
        xp.when('跳过').click()
        xp.when('允许').click()
        xp('立即体验').click()
        logger.info('Search')
        xp('搜索').click()
        s.send_keys('周杰伦')
        s.send_action('search')
        self.assertTrue(xp('布拉格广场').wait())

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        d.set_fastinput_ime(False)
        d.app_stop(self.package_name)
        d.screen_off()
if __name__ == '__main__':
    unittest.main()