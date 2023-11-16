import requests
from bs4 import BeautifulSoup
import json


def findInList(value, list):
    result = [i for i in list if i['code'] == str(value)]
    return result[0] if len(result) > 0 else {}

def getAlternative(code):
    url = "https://www.truper.com/restDataSheet/api/search/products.php"
    response = requests.post(url, headers={}, data={'word': code})
    if(response.status_code==200):
        data = json.loads(response.content)
    if len(data) > 1:
        product = [item for item in data if item['code'] == code]
        if len(product) == 0:
            return None
    elif len(data) == 1:
        product = data
    else:
        return None
    url_id = product[0]['url']
    d = requests.get(url_id, headers={}, data={})
    soup = BeautifulSoup(d.text, 'html.parser')
    input = soup.find(id="dataSheetId")
    if input:
        id = input['idproduct']
    else:
        id = None
    return id

def getMain(code):
    url = "https://www.truper.com/restDataSheet2/api/products/searchDownloads.php"
    response = requests.post(url, headers={}, data={'word': code})
    if(response.status_code==200):
        data = json.loads(response.content)
        if len(data) < 1:
            return None
        else:
            data = data['data']
        if len(data) > 1:
            product = findInList(value=code,list = data)
            if len(product) > 1:
                id = product['id']
            else:
                id = None
        elif len(data) == 1:
            id = data[0]['id']
        else:
            id = None
    else:
        id = None
    return id

def getIdFromCode(code):
    id = getMain(code)
    if id:
        return id
    id =  getAlternative(code)
    if id:
        return id
    return None