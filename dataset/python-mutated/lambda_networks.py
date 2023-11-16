from urllib.request import Request, urlopen

def handler(event, context):
    if False:
        return 10
    url = event.get('url')
    httprequest = Request(url, headers={'Accept': 'application/json'})
    with urlopen(httprequest) as response:
        return {'status': response.status, 'response': response.read().decode()}