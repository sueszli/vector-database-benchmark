from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time, sys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
reload(sys)
sys.setdefaultencoding('utf-8')
Type = sys.getfilesystemencoding()

def main():
    if False:
        print('Hello World!')
    driver = webdriver.Chrome()
    now_url = 'https://www.taobao.com/'
    login_url = 'https://login.taobao.com/member/login.jhtml'
    driver.get(now_url)
    name = 'temp.png'
    driver.save_screenshot(name)
    newwindow = 'window.open("https://www.taobao.com/");'
    driver.delete_all_cookies()
    driver.add_cookie({'cookie': 'thw=cn; _med=dw:1280&dh:800&pw:2560&ph:1600&ist:0; cna=P2IsEanwSRwCAQHAWqiGF67j; v=0; _tb_token_=eee3b5ee33fe; uc1=cookie14=UoW%2FWvN6rMfscA%3D%3D&lng=zh_CN&cookie16=UIHiLt3xCS3yM2h4eKHS9lpEOw%3D%3D&existShop=false&cookie21=U%2BGCWk%2F7pY%2FF&tag=7&cookie15=UIHiLt3xD8xYTw%3D%3D&pas=0; uc3=sg2=BYiIfEpsMbxtm040yzQn62r4dy8462CfLR73vjezc00%3D&nk2=AimQPFamtydz&id2=UUkKfSsJrCYO&vt3=F8dARHtAw8YORZlfWNE%3D&lg2=UIHiLt3xD8xYTw%3D%3D; hng=CN%7Czh-cn%7CCNY; existShop=MTQ4NzIzODY0Mw%3D%3D; uss=UUo3ufYf5xKnNsaX1Did8zEif4JWaXQKBqHBcNPFsBnDoRjsJJLEk3H3; lgc=a83533774; tracknick=a83533774; cookie2=1ce7c2d2ca6f9c4ae2d6572991049a5c; sg=450; mt=np=; cookie1=VFRzDaFMVd2CkhbafcMIU%2FP3OBRn%2FPsNhKwkjUH18W0%3D; unb=214163505; skt=c6537c77a7d1eeab; t=7770838b844b0e51e306c6d6ea1afd1d; publishItemObj=Ng%3D%3D; _cc_=VFC%2FuZ9ajQ%3D%3D; tg=0; _l_g_=Ug%3D%3D; _nk_=a83533774; cookie17=UUkKfSsJrCYO; l=AiIincM6w0qeIuxe-pmmoAbh8qKF/yZm; isg=AiAgn3SvbMYgs9DCPrh-kuXo8Sh8CgTzgPI-RZowYDsllcK_QjisgmnlW4rv'})
    driver.execute_script(newwindow)
    input('查看效果')
    driver.quit()
    driver.close()
if __name__ == '__main__':
    main()