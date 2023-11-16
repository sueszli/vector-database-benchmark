from requests_html import HTMLSession
'\ninstalar  libreria pip install  requests_html\n\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    session = HTMLSession()
    page = session.get('https://holamundo.day')
    '\n       al inspecionar la pagina puedes ver que cada  evento de la \n       charla usa una etiqueta blockquote solo hay que ver cuantos elementos son \n       y traer el rango en este caso [19:34]\n       postd: los conte de manera manual para saber el rango  debe de haber una mejor manera\n       de hacerlo\n    '
    for event in page.html.find('blockquote')[19:34]:
        print(event.text)
if __name__ == '__main__':
    main()