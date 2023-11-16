import requests
from bs4 import BeautifulSoup as bs
url = 'https://holamundo.day/'
site = requests.get(url)
soup = bs(site.content, 'html.parser')

class RetoWebScrapping:

    def __init__(self, url: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n\n        Args:\n            url (str): url to the website to scrap\n        '
        self.site = requests.get(url)
        self.soup = bs(site.content, 'html.parser')
        self.agenda = {}

    def find(self, day: int):
        if False:
            i = 10
            return i + 15
        'Function to search for events on a given day\n        Args:\n            day (int): \n        '
        for e in self.soup.find_all('h1'):
            title = e
            n = title.find_next_sibling('blockquote')
            evento = title.text
            if evento.find(str(day)) != -1:
                print(evento)
                while n != None:
                    evento = n.text
                    evento = evento.split(' | ', 1)
                    self.agenda[evento[0]] = evento[1]
                    n = n.find_next_sibling('blockquote')
                self.printAgenda(self.agenda)

    def printAgenda(self, agenda):
        if False:
            print('Hello World!')
        for (h, v) in agenda.items():
            print('Hora: {} --- {}'.format(h, v))
        pass
reto = RetoWebScrapping(url)
reto.find(8)