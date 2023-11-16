from bs4 import BeautifulSoup
import requests

class Horarios:

    def __init__(self, url):
        if False:
            i = 10
            return i + 15
        self.url = url

    def url_go(self):
        if False:
            print('Hello World!')
        try:
            link = requests.get(self.url)
            link.raise_for_status()
            return link.content
        except requests.exceptions.RequestException as e:
            print(e)
            return None

    def listado_horario(self, soup) -> list:
        if False:
            for i in range(10):
                print('nop')
        busca = soup.find_all(string=' Agenda 8 de mayo | ')[0].find_parent('h1')
        lista = []
        for s in busca.find_all_next('blockquote'):
            lista.append(s.get_text())
        return lista

    def print_horarios(self, horarios):
        if False:
            return 10
        print('Agenda 8 de mayo | “Hola Mundo” day')
        for horario in horarios:
            print(horario)

    def start(self):
        if False:
            print('Hello World!')
        content = self.url_go()
        soup = BeautifulSoup(content, 'html.parser')
        horarios = self.listado_horario(soup)
        self.print_horarios(horarios)
HolaMundoDay = Horarios('https://holamundo.day/')
HolaMundoDay.start()