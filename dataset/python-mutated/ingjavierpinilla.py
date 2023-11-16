import requests
from bs4 import BeautifulSoup

def get_html(url: str) -> str:
    if False:
        return 10
    if url is None:
        raise SystemExit() from None
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err) from None
    return response.text

def get_schedule(doc: str, text: str=' Agenda 8 de mayo | ') -> list:
    if False:
        for i in range(10):
            print('nop')
    title = doc.find_all(string=text)[0].find_parent('h1')
    return [appointment.get_text() for appointment in title.find_all_next('blockquote')]

def main():
    if False:
        while True:
            i = 10
    url = 'https://holamundo.day/'
    doc = BeautifulSoup(get_html(url), 'html.parser')
    appointments = get_schedule(doc)
    for appointment in appointments:
        print(appointment)
if __name__ == '__main__':
    main()