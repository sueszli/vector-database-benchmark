import uiautomator2 as u2
import adbutils
import threading
from logzero import logger

def worker(d: u2.Device):
    if False:
        for i in range(10):
            print('nop')
    d.app_start('io.appium.android.apis', stop=True)
    d(text='App').wait()
    for el in d.xpath('@android:id/list').child('/android.widget.TextView').all():
        logger.info('%s click %s', d.serial, el.text)
        el.click()
        d.press('back')
    logger.info('%s DONE', d.serial)
for dev in adbutils.adb.device_list():
    print('Dev:', dev)
    d = u2.connect(dev.serial)
    t = threading.Thread(target=worker, args=(d,))
    t.start()