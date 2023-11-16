from requests_html import HTMLSession
import re
URL = 'https://holamundo.day/'

def get_agenda(n_agenda: int):
    if False:
        while True:
            i = 10
    session = HTMLSession()
    web_data = session.get(URL).html.find('.BlockquoteElement___StyledBlockquote-sc-1dtx4ci-0')
    count = 0
    for element in web_data:
        try:
            valid_content = re.match('[\\d:|]', element.text).string
            if re.search('Bienvenida', valid_content):
                count += 1
            if count == n_agenda:
                print(valid_content)
        except AttributeError:
            pass
if __name__ == '__main__':
    get_agenda(2)