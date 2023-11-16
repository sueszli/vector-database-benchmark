from requests import get
from os import system, getcwd, path
url = 'https://source.unsplash.com/random'
filename = 'random.jpg'

def download(url, file_name):
    if False:
        return 10
    '\n    downloading the file and saving it\n    '
    with open(file_name, 'wb') as file:
        response = get(url)
        file.write(response.content)

def setup(pathtofile):
    if False:
        while True:
            i = 10
    '\n    setting the up file\n    '
    system('nitrogen --set-auto {}'.format(path.join(getcwd(), pathtofile)))
if __name__ == '__main__':
    download(url, filename)
    setup(filename)