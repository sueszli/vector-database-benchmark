import requests
from bs4 import BeautifulSoup

def print_day_8():
    if False:
        return 10
    url = 'https://holamundo.day'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    results = soup.find_all('h1')
    for result in results:
        if result.text.find('Agenda 8 de mayo') != -1:
            print(result.text)
            sibling = result.next_siblings
            for agenda in sibling:
                if agenda.name == 'blockquote':
                    print(agenda.text)
print_day_8()