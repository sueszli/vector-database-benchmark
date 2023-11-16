import requests
from bs4 import BeautifulSoup

def Titulo(tag):
    if False:
        return 10
    return tag.name == 'h1' and 'Agenda 8 de mayo' in tag.text

def ResponseAgenda(url):
    if False:
        i = 10
        return i + 15
    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    Fecha = soup.find(Titulo)
    Horario = Fecha.find_all_next('blockquote', {'class': 'BlockquoteElement___StyledBlockquote-sc-1dtx4ci-0 slate-BlockquoteElement notion-quote unset-width'})
    for Hora in Horario:
        print(Hora.get_text())
url = 'https://holamundo.day/'
ResponseAgenda(url)