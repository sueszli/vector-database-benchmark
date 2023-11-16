import requests
import os

def download_file_from_google_drive(id, destination):
    if False:
        i = 10
        return i + 15
    URL = 'https://docs.google.com/uc?export=download'
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)

def get_confirm_token(response):
    if False:
        print('Hello World!')
    for (key, value) in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    if False:
        print('Hello World!')
    CHUNK_SIZE = 32768
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
print('Dowloading Sony subset... (25GB)')
download_file_from_google_drive('10kpAcvldtcb9G2ze5hTcF1odzu4V_Zvh', 'dataset/Sony.zip')
print('Dowloading Fuji subset... (52GB)')
download_file_from_google_drive('12hvKCjwuilKTZPe9EZ7ZTb-azOmUA3HT', 'dataset/Fuji.zip')
os.system('unzip dataset/Sony.zip -d dataset')
os.system('unzip dataset/Fuji.zip -d dataset')