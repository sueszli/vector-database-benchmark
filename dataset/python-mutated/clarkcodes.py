from bs4 import BeautifulSoup
import requests
import typer
from rich import print
from rich.table import Table
URL = 'https://holamundo.day'

def scraper():
    if False:
        while True:
            i = 10
    global URL
    title = ''
    response = requests.get(URL).content
    bs = BeautifulSoup(response, features='html.parser')
    titles = bs.find_all('h1', 'StyledElement___StyledDiv-sc-2e063k-0 notion-h notion-h1 unset-width')
    blockquotes = bs.find_all('blockquote')
    for t in titles:
        if str(t.text).find('>_ Agenda 8') != -1:
            title = t.text
            break
    print(f'[bold yellow]\n{title}\n')
    scheduleTable = Table('Hora', 'Descripción: Tema | Ponente')
    for bq in blockquotes[21:]:
        bqText = str(bq.text)
        separatorIndex = bqText.find(' | ')
        scheduleTable.add_row(bqText[0:separatorIndex], bqText[separatorIndex + 3:])
    print(scheduleTable)

def main():
    if False:
        for i in range(10):
            print('nop')
    print('[bold green]\n*** Reto #18: WEB SCRAPING - By @ClarkCodes ***')
    print('[bold green]Web Scraping with Python')
    scraper()
    print('[green]\nEsto ha sido todo por hoy.\nMuchas gracias por ejecutar este Script, hasta la próxima... Happy Coding!, bye :D\nClark.')
if __name__ == '__main__':
    typer.run(main)