from robot.api.deco import keyword, library, not_keyword
from os.path import abspath
not_keyword(abspath)

def exposed_in_module():
    if False:
        return 10
    pass

@not_keyword
def not_exposed_in_module():
    if False:
        return 10
    pass

@keyword
@not_keyword
def keyword_and_not_keyword():
    if False:
        while True:
            i = 10
    pass

def not_exposed_by_setting_attribute():
    if False:
        for i in range(10):
            print('nop')
    pass
not_exposed_by_setting_attribute.robot_not_keyword = True